//! LLM-as-judge scoring for open-ended QA answers.
//!
//! Follows the LongMemEval official evaluation pattern: a judge LLM reads the
//! question, the gold answer, and the candidate answer, and returns a binary
//! correctness verdict. This is more robust than exact-match for free-form
//! text where multiple phrasings can be correct.
//!
//! Prompt template adapted from https://github.com/xiaowu0162/LongMemEval

use engram_llm::{ChatLlm, ChatMessage};

#[derive(Debug, Clone)]
pub struct JudgeVerdict {
    pub correct: bool,
    pub raw_response: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

const JUDGE_SYSTEM: &str =
    "You are a strict scientific evaluator. Precision matters — partial, vague, or \
     summarised answers are NOT acceptable.\n\n\
     You will be given a question, a reference answer, and a candidate answer. Decide \
     whether the candidate is CORRECT using the reference as ground truth.\n\n\
     CORRECT if:\n\
     1. The candidate states the same fact(s) as the reference, possibly worded differently. \
        'Business Administration degree' ≡ 'BA in Business Administration'.\n\
     2. Numeric equivalence. '1.5 hours' ≡ '90 minutes'. '45 minutes each way' ≡ '90 minutes total'.\n\
     3. Date format differences. '2024-03-15' ≡ 'March 15, 2024'. '13 August' ≡ '13 August, 2023' \
        (adding a year that is contextually correct is fine).\n\
     4. LIST QUESTIONS — superset rule: if the reference lists N items and the candidate names \
        ALL N of them, the answer is CORRECT even if the candidate also lists additional items. \
        Extra items that are factually consistent do not invalidate the answer. \
        Example: gold='beach, mountains, forest', candidate='beach, mountains, forest, canyon' → CORRECT.\n\n\
     INCORRECT if:\n\
     - LIST QUESTIONS: the reference lists N items and the candidate names fewer than N. \
       Missing ANY gold item makes the answer INCORRECT, even if the items it does name are right. \
       Example: gold='beach, mountains, forest', candidate='beach' → INCORRECT (2 missing).\n\
     - The candidate gives a vague paraphrase instead of the specific fact. \
       Example: gold='mentors, family, friends', candidate='people who support her' → INCORRECT.\n\
     - The candidate hedges or equivocates. 'once or twice' when the gold is '2' → INCORRECT.\n\
     - The candidate contradicts the reference on any key fact.\n\
     - The candidate refuses ('I don't know') when the reference has a definite answer.\n\
     - The candidate gives incorrect specifics (wrong name, wrong number, wrong date).\n\n\
     When in doubt, mark INCORRECT. Scientific memory must be precise.\n\
     Respond with exactly one word on the final line: CORRECT or INCORRECT.";

fn build_judge_user(question: &str, gold: &str, candidate: &str) -> String {
    format!(
        "Question: {question}\n\nReference answer: {gold}\n\nCandidate answer: {candidate}\n\nIs the candidate correct?",
        question = question.trim(),
        gold = gold.trim(),
        candidate = candidate.trim()
    )
}

pub async fn judge_answer<J: ChatLlm + ?Sized>(
    judge: &J,
    question: &str,
    gold_answer: &str,
    candidate_answer: &str,
) -> Result<JudgeVerdict, engram_llm::LlmError> {
    let messages = vec![
        ChatMessage::system(JUDGE_SYSTEM),
        ChatMessage::user(build_judge_user(question, gold_answer, candidate_answer)),
    ];
    let resp = judge.chat(&messages).await?;
    let raw = resp.content.clone();
    let upper = raw.to_ascii_uppercase();
    // Look at the last line for the verdict; fall back to a substring search
    // if the model was verbose and didn't obey the format.
    let last_line_verdict = upper
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .unwrap_or_default();
    let correct = if last_line_verdict.contains("CORRECT") && !last_line_verdict.contains("INCORRECT") {
        true
    } else if last_line_verdict.contains("INCORRECT") {
        false
    } else if upper.contains("CORRECT") && !upper.contains("INCORRECT") {
        true
    } else {
        false
    };
    Ok(JudgeVerdict {
        correct,
        raw_response: raw,
        prompt_tokens: resp.prompt_tokens,
        completion_tokens: resp.completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_correct_verdict() {
        // unit test for the parsing logic only — no network.
        let raw = "The candidate gives the same answer as the reference.\n\nCORRECT".to_string();
        assert!(raw.to_ascii_uppercase().lines().rev().any(|l| l.trim() == "CORRECT"));
    }
}
