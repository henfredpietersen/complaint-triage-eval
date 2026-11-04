# complaint-triage-eval

Real consumer complaints about banks, lenders and payment companies get sorted into the team that should own them. When the model isn't sure, it hands the complaint to a person instead of guessing. It's scored on complaints it never saw in training, and the mistakes are shown, not hidden.

**Try it live:** https://outview.co.za/case-study-complaint-triage/ (trains and scores in your browser, takes a few seconds to load)

Built AI-assisted under my direction. I picked the data, the hand-over rule and how it's scored. Claude generated the code, and I reviewed it, ran it and checked the tests. I didn't hand-write it.

## What it does

1. Takes a complaint in the consumer's own words.
2. Scores it against six products (checking/savings, credit card, debt collection, money transfer, mortgage, vehicle loan) and picks the most likely one.
3. If the top score is 60% or more, it routes the complaint to that product's queue. Below that, it goes to a person.
4. Shows why: the words that pushed it there, and the three most similar past complaints, for whoever picks it up.

No large language model. It's TF-IDF word scoring and logistic regression in plain Python (standard library only), so the same file runs in CI and in the browser through Pyodide. That's on purpose: it's cheap, explainable and measurable, and anything fancier now has a number to beat.

## The data

The CFPB's [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (public domain). The CFPB stopped publishing complaint text in August 2026 and moved the text it had already published to a [FOIA archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). Two things I found while building this:

- The **August 2026** archive file has 667,140 complaints and no text at all, so it can't be used.
- The **May 2026** file has 619,940 complaints, and 17,528 of them have the consumer's text. That's the one used here.

`data/snapshot.json` holds a fixed-seed sample of 200 complaints for each of the six most common products: 1,200 in total. ZIP code and state are dropped. The CFPB had already replaced personal details with XXXX. The sample is balanced so every queue gets a fair test. The real mix is very uneven (the counts are in the snapshot and on the demo page), and the scores would shift on it.

The workflow `.github/workflows/snapshot.yml` downloads the archive file on GitHub's servers and rebuilds the snapshot (`scripts/make_snapshot.py`). The file's SHA-256 is stored with it. The CFPB's live API no longer returns complaint text and doesn't allow calls from other websites, which is why the demo uses a snapshot and not live data.

## How well it does

Trained on 845 complaints and tested on 355 it never saw (the split is a hash of the complaint ID, 30% test):

| | |
|---|---|
| Right product, all test complaints | **79.4%** |
| Always guessing the most common product | 20.9% |
| Routed automatically at the 60% threshold | **72.1%** of complaints |
| Of those, sent to the right queue | **90.2%** |
| Sent to the wrong queue | 25 of 355 |
| Handed to a person | the other 99 |

The settings (12 training passes, learning rate 2.0, the 60% threshold) were picked with `scripts/tune.py` on a slice of the training data, never on the test set. On that slice the 60% threshold gave about 93% right on routed complaints. On the test set it's 90.2%, so expect a little less than the tuning suggested.

The most common mix-up is between money-transfer and checking-account complaints: probably because money sent through an app lands in a bank account, so the two read alike. The demo page lists every mistake, most confident first.

## What this isn't

- It doesn't read for meaning. A complaint that uses another product's words will fool it.
- It doesn't write replies. It routes and shows its evidence.
- Six products only. Complaints about anything else get forced into one of the six (with the threshold, many of those go to a person).
- Complaints are what people reported, not proven facts, and the CFPB says the database isn't a statistical sample. Nothing here says anything about a particular company.

## Run it

```bash
python -m pytest -q                    # 12 tests, standard library + pytest
python scripts/tune.py                 # how the settings were picked
python web/build.py                    # rebuild docs/demo.js after changing triage/ or web/
open docs/demo.html                    # needs a local web server for the snapshot fetch:
python -m http.server -d . 8000        # then http://localhost:8000/docs/demo.html
```

## What was tested

- `tests/test_model.py`: a hand-made set where the answers are obvious. Masking and stop words are dropped, the split is stable, clear text goes to the right queue, unknown words and low confidence go to a person, training is deterministic, and the scorecard counts add up.
- `tests/test_snapshot_and_page.py`: the snapshot matches its own description; a fresh run reproduces every number quoted in this README; the published page embeds the current code; and the page's example complaints all come from the test set.
- I also ran the page's code under Pyodide 0.26.4 (the version the page loads) and got the same scores as CPython.

CI runs lint, the tests and the page drift check on every push.

## Files

```
triage/model.py          tokeniser, TF-IDF, logistic regression, abstain rule, scorecard
web/driver.py            what the page calls
web/page.js, build.py    the demo page; build.py embeds the Python into docs/demo.js
scripts/make_snapshot.py builds data/snapshot.json from a CFPB archive file
scripts/tune.py          settings search on training data only
```

MIT licence. Data: CFPB Consumer Complaint Database, public domain.
