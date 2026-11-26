//! Tiny OpenAI-compatible chat client (Gemini, LM Studio, Ollama, OpenRouter).
use crate::settings::{Settings, env_or_dotenv};
use anyhow::{Context, Result, bail};
use regex::Regex;
use serde_json::{Value, json};
use std::time::Duration;

pub struct Llm {
    base_url: String,
    api_key: String,
    pub model: String,
    /// Tried once when `model` stays overloaded (Gemini Flash returned 503 "high demand" for
    /// minutes on a real run while Flash-Lite answered in 2.5 s).
    fallback: Option<String>,
}

impl Llm {
    pub fn new(provider: &str, model: Option<&str>, vision: bool, s: &Settings) -> Result<Self> {
        // (base_url, key, default model, vision model)
        let (base, key, def, vis): (&str, Option<String>, &str, Option<&str>) = match provider {
            "gemini" => (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                Some(s.gemini_key().context("set the Gemini API key in Settings")?),
                "gemini-flash-latest",
                Some("gemini-flash-latest"),
            ),
            "lmstudio" => ("http://localhost:1234/v1", None, "local-model", None),
            "ollama" => ("http://localhost:11434/v1", None, "qwen3:14b", None),
            "openrouter" => (
                "https://openrouter.ai/api/v1",
                Some(s.openrouter_key().context("set the OpenRouter API key in Settings")?),
                "google/gemini-3.8-flash",
                None,
            ),
            other => bail!("unknown LLM provider '{other}'"),
        };
        let base_url = s.base_url.clone().filter(|u| !u.is_empty())
            .or_else(|| env_or_dotenv("CLIPFORGE_LLM_BASE_URL"))
            .unwrap_or_else(|| base.to_string());
        let api_key = env_or_dotenv("CLIPFORGE_LLM_API_KEY").or(key).unwrap_or_else(|| "none".into());
        let default_model = if vision { vis.unwrap_or(def) } else { def };
        let model = model.map(str::to_string)
            .or_else(|| s.llm_model.clone().filter(|m| !m.is_empty()))
            .or_else(|| env_or_dotenv("CLIPFORGE_LLM_MODEL"))
            .unwrap_or_else(|| default_model.to_string());
        let lite = "gemini-flash-lite-latest";
        let fallback = (provider == "gemini" && model != lite).then(|| lite.to_string());
        Ok(Self { base_url: base_url.trim_end_matches('/').to_string(), api_key, model, fallback })
    }

    pub fn chat(&self, messages: Value) -> Result<String> {
        match self.chat_with(&self.model, &messages) {
            Err(e) if e.downcast_ref::<Busy>().is_some() && self.fallback.is_some() => {
                let fb = self.fallback.as_deref().unwrap_or_default();
                self.chat_with(fb, &messages).with_context(|| format!("{} is overloaded ({e}); {fb} also failed", self.model))
            }
            r => r,
        }
    }

    fn chat_with(&self, model: &str, messages: &Value) -> Result<String> {
        // A ranking reply takes seconds; an overloaded model can instead hang for minutes (seen:
        // 180 s then a timeout), so don't wait long before trying the fallback model.
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(75))
            .build()?;
        let body = json!({ "model": model, "messages": messages, "temperature": 0.4 });
        // Busy or rate-limited providers usually recover within seconds; retry those briefly.
        let waits = [3, 8];
        let mut attempt = 0;
        loop {
            crate::cancel::check()?;
            let resp = match client.post(format!("{}/chat/completions", self.base_url)).bearer_auth(&self.api_key).json(&body).send() {
                Ok(r) => r,
                // no answer at all (timeout, refused, reset): same as busy for the fallback
                Err(e) if e.is_timeout() || e.is_connect() => return Err(Busy(format!("{model}: {e}")).into()),
                Err(e) => return Err(e.into()),
            };
            let status = resp.status();
            let text = resp.text()?;
            if status.is_success() {
                let data: Value = serde_json::from_str(&text)?;
                return Ok(data["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string());
            }
            let transient = matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504);
            match waits.get(attempt) {
                Some(w) if transient => {
                    std::thread::sleep(Duration::from_secs(*w));
                    attempt += 1;
                }
                _ if transient => return Err(Busy(format!("{model}: HTTP {}", status.as_u16())).into()),
                _ => bail!("LLM HTTP {}: {}", status.as_u16(), text.chars().take(600).collect::<String>()),
            }
        }
    }
}

/// The provider kept answering "busy" (429/5xx) after the retries.
#[derive(Debug)]
struct Busy(String);

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} after retries", self.0)
    }
}

impl std::error::Error for Busy {}

/// Pull the first JSON object/array out of a model reply (handles ```json fences, <think> blocks).
pub fn parse_json(text: &str) -> Result<Value> {
    let think = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let text = think.replace_all(text, "");
    let text = text.replace("```json", "").replace("```", "");
    let start = [text.find('['), text.find('{')].into_iter().flatten().min().context("no JSON in reply")?;
    let closer = if text[start..].starts_with('[') { ']' } else { '}' };
    let end = text.rfind(closer).context("unterminated JSON in reply")?;
    Ok(serde_json::from_str(&text[start..=end])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_model_wins() {
        let s = Settings { gemini_api_key: "test-key".into(), ..Settings::default() };
        let llm = Llm::new("gemini", Some("my-model"), false, &s).unwrap();
        assert_eq!(llm.model, "my-model");
        assert!(Llm::new("nope", None, false, &s).is_err());
    }

    #[test]
    fn parse_json_strips_fences_and_think() {
        let v = parse_json("<think>hm</think>```json\n[{\"start\": 10.0}]\n```").unwrap();
        assert_eq!(v[0]["start"], 10.0);
        let v = parse_json("Sure: {\"clips\": []} done").unwrap();
        assert!(v["clips"].is_array());
    }
}
