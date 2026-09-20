# Re-checking Amazon URLs

CLAUDE.md §5.1: *"`url` must resolve — a judge will click one."*

The Amazon corpus comes from a 2023 review dump. Amazon delists products, so the
dataset rots: **31% of the ASINs the first build selected already 404'd.** Dead
links have to be found and excluded, or the demo invites a judge to click one.

Run this before demo day, and after any corpus rebuild.

## Why not curl

You cannot check this from the command line. Amazon serves an **identical
3,790-byte bot-block page** for live and dead ASINs alike — same 200 status, same
body. A shell check will tell you every URL is fine.

```
B0732SGQ7M  HTTP:200  bytes=3790   <- live product
B09W66MSPX  HTTP:200  bytes=3790   <- 404 page, indistinguishable
```

The check has to run inside a real browser session, where Amazon serves the real
page.

## The procedure

1. Open `https://www.amazon.com` in a browser and open the devtools console.
   Same-origin `fetch` from that tab carries the session, so Amazon responds
   normally.

2. Paste the checker:

```js
window.__check = async (asins, conc = 10) => {
  const dead = [], live = [], err = [];
  let i = 0;
  const worker = async () => {
    while (i < asins.length) {
      const a = asins[i++];
      try {
        const r = await fetch(`/dp/${a}`, { redirect: 'follow' });
        const t = await r.text();
        if (r.status === 200 && /id="productTitle"|id="title"/i.test(t)) live.push(a);
        else dead.push(a);
      } catch (e) { err.push(a); }
    }
  };
  await Promise.all(Array.from({ length: conc }, worker));
  return { n: asins.length, live: live.length, dead, err };
};
```

3. Get the current ASINs:

```bash
python -c "import json; rows=[json.loads(l) for l in open('data/signals.jsonl',encoding='utf-8')]; print(json.dumps(sorted({r['url'].rsplit('/',1)[-1] for r in rows if r['source']=='amazon'})))"
```

4. Run `await window.__check([...])` in batches of ~80. Larger batches risk the
   console timing out; concurrency 10–12 is comfortable without tripping rate
   limits.

5. Add anything in `dead` to the `dead` array in `data/dead_asins.json`, then
   rebuild:

```bash
python scripts/build_corpus.py --reddit-shard data/raw/<shard>.parquet
```

6. **Repeat.** The rebuild backfills from fresh candidates, which introduces
   ASINs nobody has checked. Iterate until a run reports zero dead. It took
   three passes to converge the first time (217 ASINs checked, 69 denylisted).

## Status

Last verified **2026-09-20**: all 152 Amazon ASINs in the corpus resolve to a
live product page. 69 dead ASINs are denylisted in `data/dead_asins.json`.

Reddit permalinks need no equivalent check — comment URLs are permanent, and a
deleted comment still resolves to its thread rather than 404ing.
