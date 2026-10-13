# tallgrass_small_docs_qa

A small question-answering (RAG) agent over a folder of messy documents from a fictional home care agency, Tallgrass Home Care. All documents are made up practice data.

## Corpus

The documents live in `data/corpus/`. Each one is messy in a different way, so each needs its own reading and cleaning step before it can be split into passages and indexed.

| File | Type | What is in it | What is needed to clean it |
|---|---|---|---|
| `caregiver_handbook.md` | Markdown | Scheduling, clock-in, pay, PTO, medication, incident and training rules (January 2026) | Remove the `Page N of 4` footers in the middle of the text, keeping the page number as metadata. Split on `##` / `###` headings and add the section path to each passage. |
| `service_rates_2026.csv` | CSV table | Hourly rates for 8 services, effective 2026-01-01 | Skip the title lines and find the real header on row 4. Keep the effective date and the holiday-rate footer note as context. Turn each row into a sentence-like passage. Mark the exceptions: Live-In is a daily rate, and Transportation has client mileage at $0.85 per mile. |
| `staff_newsletter_spring_2026.pdf` | PDF, two columns | Clock-in app rollout, Caregiver of the Quarter, training dates, referral bonus | A plain text read mixes the two columns line by line. Use word positions to detect the layout, pull out the full-width title and footer, then read the left column (x around 54) before the right column (x around 320). Split on article headings. |
| `scanned_memo_snow_days.pdf` | PDF, scanned image | Snow day policy memo dated November 3, 2025 | There is no text layer, so it needs OCR (for example tesseract). Deskew and convert to black and white first. Store the To/From/Date/Re fields as metadata, and mark the passages as OCR text. |

### Steps for all documents

- Detect the problem pages automatically: two-column pages go to column repair, and pages with no text but an image go to OCR.
- Normalize the text: plain quotes and dashes, collapsed whitespace, rejoined hyphenated words. Keep dollar amounts, decimals and times exactly as written.
- Split into passages of at most 150 words on section boundaries (see [Chunking strategy](#chunking-strategy)). Each passage carries `source_file`, `page`, `section`, `extraction_method` and `doc_date`, so answers can cite their source.
- Keep document dates, because some answers depend on timing (for example, the newsletter's March 9 VisitTrack launch).
- Retrieve enough passages (k of 3 to 5) for questions that span files, such as in-service training hours, which appear in both the handbook and the newsletter.

## Preparing the corpus

Requires [uv](https://docs.astral.sh/uv/) and tesseract (`brew install tesseract`).

```bash
uv sync
uv run doc-qa-prepare            # reads data/corpus, writes data/processed
```

This writes to `data/processed/`:

- `<doc_id>.md`: the cleaned text of each document, by section, with page numbers.
- `passages.jsonl`: one passage per line with `passage_id`, `doc_id`, `source_file`, `title`, `page`, `section`, `text`, `extraction_method` (`text`, `column_repair`, `ocr`, `csv_row` or `csv_note`) and `doc_date`.
- `documents.json`: per-document metadata (for example the memo's To/From/Date/Re fields) and page checks (two-column, scanned, OCR confidence).

## Chunking strategy

Passages follow each document's own structure (headings, table rows, memo body) instead of a fixed character window. Each reader first produces blocks, one per section, and `passages.py` then splits any block that is too long:

- The limit is 150 words per passage. There is no minimum, so a short section stays a short passage rather than being merged with an unrelated one.
- A long block is split at paragraph boundaries. A paragraph that is longer than the limit on its own is split between sentences, and the last sentence of one passage is repeated at the start of the next. A sentence is never cut, so answer phrases stay whole.
- The section name is stored in the `section` field, not in `text`. The indexing step should put it in front of the text, because headings like "Pay > Mileage" help retrieval.

| Document | Split on | Passages | Notes |
|---|---|---|---|
| `caregiver_handbook.md` | `##` and `###` headings | 10 | Section is the heading path, for example `Pay > Overtime`. Page comes from the removed `Page N of 4` footers. The intro lines (fictional-agency note, "Last updated") form their own passage under the document title. |
| `service_rates_2026.csv` | One passage per table row | 10 | 8 row passages, plus one for the title lines and one for the holiday-rate note. Each row passage starts with the table title and "Rates are USD per hour unless the notes say otherwise", then lists `column: value` pairs with the original column names, so a row makes sense on its own. Section is `code - name`, for example `RS-03 - Respite Care`. |
| `staff_newsletter_spring_2026.pdf` | Article headings (bold text larger than the body) | 6 | Split after the column repair, so each article is one passage in reading order. The masthead and the footer line are separate passages under the newsletter title. |
| `scanned_memo_snow_days.pdf` | Whole memo | 1 | The memo is about 110 words, so it stays one passage that includes the To/From/Date/Re lines. Section is the Re line. `extraction_method` is `ocr`, so answers can warn about possible OCR errors. |

In this corpus no section goes over 150 words (the longest passage is 136), so every section is exactly one passage and the sentence overlap never triggers. It only matters if longer documents are added.

## Search

There are three search methods over `passages.jsonl`, in `src/doc_qa/search.py` and `src/doc_qa/embedders.py`. All of them search the section heading together with the passage text.

- **`bm25` (keyword search)**, implemented directly (k1 = 1.5, b = 0.75). Text is lowercased, common stopwords are dropped, and a trailing plural "s" is removed. Numbers like `38.50`, `555-0142` and `$0.67` stay as single tokens so exact figures can match.
- **`local-embedding`** with [fastembed](https://github.com/qdrant/fastembed) and `BAAI/bge-small-en-v1.5`: a 384-dimension model that runs locally, with no API key and a roughly 70 MB download on first use.
- **`api-embedding`** with `downtown-miami/openrouter/perplexity/pplx-embed-v1-0.6b`, called through the 4Geeks LiteLLM proxy (`LLM_BASE_URL`, OpenAI-compatible `/v1/embeddings`). This is the same proxy and model as the healthcore_ft_ai_1 repo. The proxy allows about 15 requests a minute, so passages and questions are sent in batches, and a 429 response waits until the limit resets.

Both embedding methods rank passages by cosine similarity. Passage and question vectors are cached per model in `data/processed/` (`embeddings_<model>_<hash>.npy`, `query_embeddings_<model>.json`, both gitignored), so the proxy is only called for text it has not embedded before.

### API setup

Copy `.example.env` to `.env` and set `LLM_API_KEY`; `.env` is gitignored. The same file sets `GENERATION_MODEL` (`downtown-miami/openrouter/deepseek/deepseek-v4-flash`, DeepSeek through the same proxy) for the answer step, which is not built yet. `LLM_SSL_VERIFY` defaults to false because the proxy's certificate chain is sometimes expired. Without a key, `--method all` skips `api-embedding`.

```bash
uv run doc-qa-search "How much is mileage reimbursed?"          # all methods, top 3
uv run doc-qa-search "weekend respite rate" --method bm25 -k 5
uv run doc-qa-eval --method all -k 1 3 5 --report results/BM25_bge-small-en-v1.5_pplx-embed-v1-0.6b.md
```

`doc-qa-eval` scores the search step against `memory_bank/references/questions.jsonl`. It reports Recall@k, Precision@k and MRR. A passage is relevant when it comes from the expected file and contains every evidence phrase. The no-answer question is listed with its top score but not scored. With `--report`, it also writes a Markdown report with per-kind and per-question results: see [results/BM25_bge-small-en-v1.5_pplx-embed-v1-0.6b.md](results/BM25_bge-small-en-v1.5_pplx-embed-v1-0.6b.md). An earlier BM25 vs bge-small report is in [results/BM25_BAAI_bge-small-en-v1.5.md](results/BM25_BAAI_bge-small-en-v1.5.md).

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 | MRR |
|---|---|---|---|---|---|---|---|
| BM25 | 0.938 | 0.938 | 0.938 | 0.938 | 0.312 | 0.188 | 0.938 |
| Embedding (bge-small-en-v1.5, local) | 0.875 | 0.938 | 0.938 | 0.875 | 0.312 | 0.188 | 0.906 |
| Embedding (pplx-embed-v1-0.6b, proxy) | 0.938 | 1.000 | 1.000 | 0.938 | 0.333 | 0.200 | 0.969 |

Each question has exactly one relevant passage, so Recall@k is the hit rate and the best possible Precision@k is 1/k (0.333 at k=3, 0.200 at k=5).

- **q06**, "What is the most PTO I can have saved up?": BM25 and bge-small both miss it. The handbook says "paid time off" and never "PTO", so BM25 has no shared word, and bge-small ranks the passage 18th of 27. pplx-embed ranks it first.
- **q14** (dementia workshop hours): bge-small ranks it second, behind the handbook's training section.
- **q13** (who to call if the app won't install): pplx-embed ranks it second.
- **q17** (dress code, no answer in the corpus): pplx-embed's top score is 0.489. That is above the 0.38 minimum score the healthcore repo uses to decide whether anything relevant was found, so that threshold needs re-tuning for this corpus before it can drive "not found" answers.

## Code

The code is in `src/doc_qa/`: one reader per file type (`markdown_reader.py`, `table_reader.py`, `pdf_reader.py` with column repair, `ocr.py`), plus `passages.py` for splitting, `prepare.py` for the prepare command, `search.py` for the search methods, `embedders.py` (local and proxy embedding models), `config.py` (proxy settings from `.env`), and `search_cli.py` / `evaluate.py` for the search and scoring commands.

The full implementation plan is in [memory_bank/references/corpus_prep_plan.md](memory_bank/references/corpus_prep_plan.md).
