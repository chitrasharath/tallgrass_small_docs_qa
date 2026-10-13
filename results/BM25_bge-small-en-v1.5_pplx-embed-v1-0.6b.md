# Search results: BM25 vs Embedding (BAAI/bge-small-en-v1.5) vs Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b)

Run on 2026-10-12 with `uv run doc-qa-eval --method all -k 1 3 5 --report <this file>`.

- Questions: `memory_bank/references/questions.jsonl` (16 scored, 1 with no answer in the corpus, not scored)
- Passages: `data/processed/passages.jsonl` (27 passages)
- A passage is relevant when it comes from the expected source file and contains every evidence phrase.
- Relevant passages per question: 1. With one relevant passage per question, Recall@k equals the hit rate and the best possible Precision@k is 1/k (1.000 at k=1, 0.333 at k=3, 0.200 at k=5).

## Summary

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 | MRR |
|---|---|---|---|---|---|---|---|
| BM25 | 0.938 | 0.938 | 0.938 | 0.938 | 0.312 | 0.188 | 0.938 |
| Embedding (BAAI/bge-small-en-v1.5) | 0.875 | 0.938 | 0.938 | 0.875 | 0.312 | 0.188 | 0.906 |
| Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) | 0.938 | 1.000 | 1.000 | 0.938 | 0.333 | 0.200 | 0.969 |

## Recall by question kind

| Method | Kind | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|---|
| BM25 | text (6) | 0.833 | 0.833 | 0.833 |
| BM25 | table (4) | 1.000 | 1.000 | 1.000 |
| BM25 | two_column (5) | 1.000 | 1.000 | 1.000 |
| BM25 | scanned (1) | 1.000 | 1.000 | 1.000 |
| Embedding (BAAI/bge-small-en-v1.5) | text (6) | 0.833 | 0.833 | 0.833 |
| Embedding (BAAI/bge-small-en-v1.5) | table (4) | 1.000 | 1.000 | 1.000 |
| Embedding (BAAI/bge-small-en-v1.5) | two_column (5) | 0.800 | 1.000 | 1.000 |
| Embedding (BAAI/bge-small-en-v1.5) | scanned (1) | 1.000 | 1.000 | 1.000 |
| Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) | text (6) | 1.000 | 1.000 | 1.000 |
| Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) | table (4) | 1.000 | 1.000 | 1.000 |
| Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) | two_column (5) | 0.800 | 1.000 | 1.000 |
| Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) | scanned (1) | 1.000 | 1.000 | 1.000 |

## Per question

Rank of the first relevant passage in the top 5 ("miss" when it is not there).

| Id | Kind | Question | Relevant passage | BM25 | Embedding (BAAI/bge-small-en-v1.5) | Embedding (downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b) |
|---|---|---|---|---|---|---|
| q01 | text | How much notice do I have to give if I can't make my shift? | `caregiver_handbook-002` | 1 | 1 | 1 |
| q02 | text | How close to the client's home do I need to be for the clock-in to work? | `caregiver_handbook-003` | 1 | 1 | 1 |
| q03 | text | How much am I reimbursed per mile for driving between clients? | `caregiver_handbook-006` | 1 | 1 | 1 |
| q04 | text | Am I allowed to give a client their pills? | `caregiver_handbook-008` | 1 | 1 | 1 |
| q05 | text | How many hours of training does every caregiver need each year? | `caregiver_handbook-010` | 1 | 1 | 1 |
| q06 | text | What is the most PTO I can have saved up? | `caregiver_handbook-007` | miss | miss | 1 |
| q07 | table | What is the weekend rate for respite care? | `service_rates_2026-004` | 1 | 1 | 1 |
| q08 | table | What is the shortest visit I can book for overnight care? | `service_rates_2026-007` | 1 | 1 | 1 |
| q09 | table | How much does live-in care cost on a weekday? | `service_rates_2026-008` | 1 | 1 | 1 |
| q10 | table | How are visits on a holiday billed? | `service_rates_2026-010` | 1 | 1 | 1 |
| q11 | two_column | When does the VisitTrack app go live? | `staff_newsletter_spring_2026-002` | 1 | 1 | 1 |
| q12 | two_column | Until what date will paper timesheets still be accepted? | `staff_newsletter_spring_2026-002` | 1 | 1 | 1 |
| q13 | two_column | Who do I call if the new app won't install on my phone? | `staff_newsletter_spring_2026-002` | 1 | 1 | 2 |
| q14 | two_column | How many in-service hours does the dementia care workshop count for? | `staff_newsletter_spring_2026-004` | 1 | 2 | 1 |
| q15 | two_column | When is the referral bonus paid out? | `staff_newsletter_spring_2026-005` | 1 | 1 | 1 |
| q16 | scanned | How much show-up pay do I get if my visit is cancelled on a Level 2 snow day? | `scanned_memo_snow_days-001` | 1 | 1 | 1 |
| q17 | no_answer | What is the dress code for caregivers? | none | not scored (top score 1.932) | not scored (top score 0.726) | not scored (top score 0.489) |
