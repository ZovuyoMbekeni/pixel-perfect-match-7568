import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import { useServerFn } from "@tanstack/react-start";
import Papa from "papaparse";
import { toast } from "sonner";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from "recharts";
import {
  analyzeTexts, extractImageText, generateInsights, SENTIMENT_MODEL,
  type SentimentLabel,
} from "@/lib/sentiment.functions";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Upload, ImageIcon, FileText, Sparkles, Download, Trash2, FlaskConical } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ToneCheck — Sentiment Analysis & Data Insights" },
      { name: "description", content: "Classify comments, CSV reviews and screenshots as positive, negative or neutral, then explore charts and generated insights." },
      { property: "og:title", content: "ToneCheck — Sentiment Analysis & Data Insights" },
      { property: "og:description", content: "Sentiment classification, interactive charts and data-grounded insights." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: App,
});

type Source = "Single" | "Multiple" | "CSV" | "Image" | "Sample";
interface Row { id: number; text: string; label: SentimentLabel; confidence: number; source: Source }

const COLORS: Record<SentimentLabel, string> = {
  Positive: "var(--positive)", Negative: "var(--negative)", Neutral: "var(--neutral)",
};
const LABELS: SentimentLabel[] = ["Positive", "Negative", "Neutral"];
const SAMPLE = [
  "The checkout process was quick and painless.",
  "My order arrived three days late and the box was damaged.",
  "The package was delivered on Tuesday.",
  "Support staff were incredibly patient and helpful!",
  "I waited on hold for over an hour. Terrible.",
  "The app works as described.",
  "Absolutely love the new design, great job.",
  "Refund still hasn't been processed after two weeks.",
];
const NAV = ["Dashboard", "Analyse Text", "Upload CSV", "Analyse Image", "Insights", "Results", "About"];
const slug = (s: string) => s.toLowerCase().replace(/\s+/g, "-");
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

function friendly(e: unknown) {
  const m = e instanceof Error ? e.message : String(e);
  if (/fetch failed|network|Failed to fetch/i.test(m)) return "Unable to connect to the sentiment-analysis service. Please try again.";
  return m;
}
function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

function App() {
  const analyze = useServerFn(analyzeTexts);
  const extract = useServerFn(extractImageText);
  const insightsFn = useServerFn(generateInsights);
  const [rows, setRows] = useState<Row[]>([]);
  const nextId = useRef(1);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [single, setSingle] = useState("");
  const [lastSingle, setLastSingle] = useState<Row | null>(null);
  const [multi, setMulti] = useState("");
  const [csvInfo, setCsvInfo] = useState<{ name: string; column: string; count: number } | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [extracted, setExtracted] = useState<string[] | null>(null);
  const [insights, setInsights] = useState<string[] | null>(null);

  async function run(texts: string[], source: Source, tag: string) {
    const clean = texts.map((t) => t.trim().slice(0, 2000)).filter(Boolean);
    if (!clean.length) { setError("Please provide some text to analyse."); return null; }
    setError(null); setBusy(tag);
    try {
      const { predictions } = await analyze({ data: { texts: clean } });
      const added = clean.map((text, i) => ({
        id: nextId.current++, text, source, label: predictions[i]!.label, confidence: predictions[i]!.confidence,
      }));
      setRows((r) => [...added, ...r]);
      setInsights(null);
      toast.success(`Analysed ${added.length} comment${added.length > 1 ? "s" : ""}`);
      return added;
    } catch (e) { setError(friendly(e)); return null; }
    finally { setBusy(null); }
  }

  const stats = useMemo(() => {
    const c = { Positive: 0, Negative: 0, Neutral: 0 } as Record<SentimentLabel, number>;
    let sum = 0;
    rows.forEach((r) => { c[r.label]++; sum += r.confidence; });
    return { total: rows.length, ...c, avg: rows.length ? sum / rows.length : 0 };
  }, [rows]);

  function onCsv(file: File) {
    setError(null);
    if (!/\.(csv|txt|tsv)$/i.test(file.name) && !/csv|text\/plain|excel/i.test(file.type)) { setError("Invalid file — please upload a .csv file."); return; }
    Papa.parse<Record<string, string>>(file, {
      header: true, skipEmptyLines: true,
      complete: async (res) => {
        const fields = res.meta.fields ?? [];
        if (!fields.length || !res.data.length) { setError("The CSV appears to be empty."); return; }
        const pref = fields.find((f) => /comment|review|text|feedback|message|content|tweet|body/i.test(f));
        const avgLen = (f: string) => res.data.reduce((s, r) => s + (r[f]?.length ?? 0), 0) / res.data.length;
        const column = pref ?? [...fields].sort((a, b) => avgLen(b) - avgLen(a))[0] ?? "";
        const texts = res.data.map((r) => r[column] ?? "").filter((t) => t.trim().length > 1);
        if (!column || !texts.length || avgLen(column) < 3) { setError("No suitable text/comment column was found in this CSV."); return; }
        if (texts.length > 500) toast.message("Only the first 500 rows will be analysed.");
        setCsvInfo({ name: file.name, column, count: Math.min(texts.length, 500) });
        await run(texts.slice(0, 500), "CSV", "csv");
      },
      error: () => setError("Could not read this CSV file."),
    });
  }

  function onImage(file: File) {
    setError(null); setExtracted(null);
    if (!/^image\/(png|jpe?g)$/.test(file.type)) { setError("Unsupported image format — please use JPG, JPEG or PNG."); return; }
    if (file.size > 4 * 1024 * 1024) { setError("Image is too large (max 4 MB)."); return; }
    const reader = new FileReader();
    reader.onload = async () => {
      const dataUrl = reader.result as string;
      setImgPreview(dataUrl); setBusy("image");
      try {
        const { lines } = await extract({ data: { dataUrl } });
        setExtracted(lines);
        setBusy(null);
        if (!lines.length) { setError("No readable text was found in this image."); return; }
        await run(lines, "Image", "image");
      } catch (e) { setError(friendly(e)); setBusy(null); }
    };
    reader.readAsDataURL(file);
  }

  async function makeInsights() {
    if (!rows.length) { setError("Analyse some comments first."); return; }
    setError(null); setBusy("insights");
    try {
      const { insights } = await insightsFn({
        data: {
          stats: { total: stats.total, positive: stats.Positive, negative: stats.Negative, neutral: stats.Neutral, avgConfidence: stats.avg },
          items: rows.slice(0, 300).map((r) => ({ text: r.text, label: r.label, confidence: r.confidence })),
        },
      });
      setInsights(insights);
    } catch (e) { setError(friendly(e)); }
    finally { setBusy(null); }
  }

  function exportCsv() {
    const csv = Papa.unparse(rows.map((r) => ({ text: r.text, sentiment: r.label, confidence: r.confidence.toFixed(4), source: r.source })));
    download("tonecheck-results.csv", csv, "text/csv");
  }
  function exportSummary() {
    const p = (n: number) => (stats.total ? ((n / stats.total) * 100).toFixed(1) : "0");
    const txt = `ToneCheck summary — ${new Date().toLocaleString()}
Sentiment model: ${SENTIMENT_MODEL}

Total comments: ${stats.total}
Positive: ${stats.Positive} (${p(stats.Positive)}%)
Negative: ${stats.Negative} (${p(stats.Negative)}%)
Neutral: ${stats.Neutral} (${p(stats.Neutral)}%)
Average confidence: ${pct(stats.avg)}

Generated insights:
${insights ? insights.map((i) => `- ${i}`).join("\n") : "(not generated)"}
`;
    download("tonecheck-summary.txt", txt, "text/plain");
  }

  const pieData = LABELS.map((l) => ({ name: l, value: stats[l] })).filter((d) => d.value);
  const confBuckets = ["50–60", "60–70", "70–80", "80–90", "90–100"].map((name, i) => ({
    name, count: rows.filter((r) => { const c = r.confidence * 100; return i === 0 ? c < 60 : c >= 50 + i * 10 && c < 60 + i * 10 + (i === 4 ? 1 : 0); }).length,
  }));

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <a href="#dashboard" className="font-display text-xl font-semibold">Tone<span className="text-primary">Check</span></a>
          <nav className="flex flex-wrap gap-1 text-sm">
            {NAV.map((n) => (
              <a key={n} href={`#${slug(n)}`} className="rounded-md px-2.5 py-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground">{n}</a>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-14 px-4 py-10">
        <section id="dashboard" className="scroll-mt-24 space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">Week 3 · Data Project</p>
              <h1 className="mt-1 text-4xl font-semibold md:text-5xl">Sentiment & Data Insights</h1>
              <p className="mt-2 max-w-xl text-muted-foreground">Every prediction below is generated live for your text.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" onClick={() => run(SAMPLE, "Sample", "sample")} disabled={!!busy}>
                <FlaskConical /> Load Sample Data
              </Button>
              {rows.length > 0 && (
                <Button variant="ghost" onClick={() => { setRows([]); setInsights(null); setLastSingle(null); }}>
                  <Trash2 /> Clear
                </Button>
              )}
            </div>
          </div>

          {error && <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</div>}
          {busy && <div className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />{busy === "image" ? "Reading text from your image..." : busy === "insights" ? "Generating insights..." : "Analysing your comments..."}</div>}

          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="Total comments" value={String(stats.total)} />
            {LABELS.map((l) => (
              <Stat key={l} label={l} value={String(stats[l])} sub={stats.total ? `${((stats[l] / stats.total) * 100).toFixed(1)}%` : "—"} color={COLORS[l]} />
            ))}
            <Stat label="Avg confidence" value={stats.total ? pct(stats.avg) : "—"} />
          </div>

          {rows.length === 0 ? (
            <div className="rounded-xl border border-dashed p-10 text-center text-muted-foreground">No data yet. Analyse text, upload a CSV or an image to populate the dashboard.</div>
          ) : (
            <div className="grid gap-4 md:grid-cols-3">
              <Panel title="Sentiment distribution">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={85} paddingAngle={2}>
                      {pieData.map((d) => <Cell key={d.name} fill={COLORS[d.name as SentimentLabel]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Panel>
              <Panel title="Sentiment counts">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={LABELS.map((l) => ({ name: l, count: stats[l] }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" fontSize={12} /><YAxis allowDecimals={false} fontSize={12} /><Tooltip />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>{LABELS.map((l) => <Cell key={l} fill={COLORS[l]} />)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Panel>
              <Panel title="Confidence levels (%)">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={confBuckets}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" fontSize={11} /><YAxis allowDecimals={false} fontSize={12} /><Tooltip />
                    <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Panel>
            </div>
          )}
        </section>

        <section id="analyse-text" className="scroll-mt-24 grid gap-6 md:grid-cols-2">
          <Panel title="Analyse one comment">
            <Textarea value={single} onChange={(e) => setSingle(e.target.value)} placeholder="Type or paste a comment..." rows={4} maxLength={2000} />
            <Button className="mt-3" disabled={!!busy} onClick={async () => { const r = await run([single], "Single", "single"); if (r?.[0]) setLastSingle(r[0]); }}>
              {busy === "single" ? <Loader2 className="animate-spin" /> : <FileText />} Analyse
            </Button>
            {lastSingle && (
              <div className="mt-4 rounded-lg bg-secondary p-4">
                <div className="text-sm text-muted-foreground">Sentiment</div>
                <div className="font-display text-2xl" style={{ color: COLORS[lastSingle.label] }}>{lastSingle.label}</div>
                <div className="text-sm">Confidence: <b>{pct(lastSingle.confidence)}</b></div>
              </div>
            )}
          </Panel>
          <Panel title="Analyse multiple comments">
            <Textarea value={multi} onChange={(e) => setMulti(e.target.value)} placeholder={"One comment per line\n..."} rows={6} />
            <Button className="mt-3" disabled={!!busy} onClick={() => run(multi.split("\n"), "Multiple", "multi")}>
              {busy === "multi" ? <Loader2 className="animate-spin" /> : <FileText />} Analyse each line
            </Button>
          </Panel>
        </section>

        <section id="upload-csv" className="scroll-mt-24 grid gap-6 md:grid-cols-2">
          <Panel title="Upload CSV">
            <p className="mb-3 text-sm text-muted-foreground">We detect the comment/review column automatically (up to 500 rows).</p>
            <FilePick accept=".csv,.txt,.tsv,text/csv,application/vnd.ms-excel" disabled={!!busy} onFile={onCsv} icon={<Upload />} label={busy === "csv" ? "Analysing..." : "Choose CSV file"} />
            {csvInfo && <p className="mt-3 text-sm">File <b>{csvInfo.name}</b> · column <b className="font-mono">{csvInfo.column}</b> · {csvInfo.count} rows</p>}
          </Panel>
          <div id="analyse-image" className="scroll-mt-24">
            <Panel title="Analyse image">
              <p className="mb-3 text-sm text-muted-foreground">JPG/PNG screenshots of comments. Text is read from the image automatically.</p>
              <FilePick accept="image/png,image/jpeg" disabled={!!busy} onFile={onImage} icon={<ImageIcon />} label={busy === "image" ? "Reading image..." : "Choose image"} />
              {imgPreview && <img src={imgPreview} alt="Uploaded" className="mt-3 max-h-48 rounded-md border object-contain" />}
              {extracted && extracted.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs font-semibold uppercase text-muted-foreground">Extracted text</div>
                  <ul className="mt-1 space-y-1 text-sm">{extracted.map((l, i) => <li key={i} className="rounded bg-secondary px-2 py-1">{l}</li>)}</ul>
                </div>
              )}
            </Panel>
          </div>
        </section>

        <section id="insights" className="scroll-mt-24">
          <div className="rounded-xl bg-ink p-6 text-ink-foreground md:p-8">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-3xl">Generated Insights</h2>
                <p className="text-sm opacity-70">Written by a separate language model from the classified results — it does not change any sentiment labels.</p>
              </div>
              <Button variant="secondary" disabled={!!busy || !rows.length} onClick={makeInsights}>
                {busy === "insights" ? <Loader2 className="animate-spin" /> : <Sparkles />} Generate insights
              </Button>
            </div>
            {insights && <ul className="mt-5 space-y-2">{insights.map((t, i) => <li key={i} className="border-l-2 border-primary pl-3">{t}</li>)}</ul>}
          </div>
        </section>

        <section id="results" className="scroll-mt-24 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-3xl">Results</h2>
            <div className="flex gap-2">
              <Button variant="outline" disabled={!rows.length} onClick={exportCsv}><Download /> Export Results</Button>
              <Button variant="outline" disabled={!rows.length} onClick={exportSummary}><Download /> Export summary</Button>
            </div>
          </div>
          <div className="overflow-x-auto rounded-xl border bg-card">
            <table className="w-full text-sm">
              <thead className="bg-secondary text-left">
                <tr><th className="p-3">Comment</th><th className="p-3">Sentiment</th><th className="p-3">Confidence</th><th className="p-3">Source</th></tr>
              </thead>
              <tbody>
                {rows.length === 0 && <tr><td colSpan={4} className="p-6 text-center text-muted-foreground">No results yet.</td></tr>}
                {rows.map((r) => (
                  <tr key={r.id} className="border-t align-top">
                    <td className="max-w-md p-3">{r.text}</td>
                    <td className="p-3"><span className="inline-flex items-center gap-1.5 font-medium"><span className="size-2 rounded-full" style={{ background: COLORS[r.label] }} />{r.label}</span></td>
                    <td className="p-3 font-mono">{pct(r.confidence)}</td>
                    <td className="p-3 text-muted-foreground">{r.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="about" className="scroll-mt-24 grid gap-6 md:grid-cols-2">
          <div>
            <h2 className="text-3xl">About the project</h2>
            <p className="mt-3 text-muted-foreground">This project demonstrates how artificial intelligence can be used to analyse text data, classify sentiment and generate useful insights from user-provided information.</p>
            <p className="mt-3 text-muted-foreground">Sentiment classification, image text reading and insights all run on the built-in language model <span className="font-mono">{SENTIMENT_MODEL}</span>. Insights are written in a separate step from classification.</p>
          </div>
          <div className="rounded-xl border bg-card p-5 text-sm">
            <h3 className="text-lg">Privacy</h3>
            <p className="mt-2 text-muted-foreground">Nothing is stored. Your text and images are sent only to the AI services for analysis and kept in this browser tab — refreshing the page clears everything. "Load Sample Data" uses a small built-in test set that is labelled as Sample, not real user data.</p>
          </div>
        </section>
      </main>
    </div>
  );
}

function Stat({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex items-center gap-1.5 text-xs uppercase tracking-wide text-muted-foreground">
        {color && <span className="size-2 rounded-full" style={{ background: color }} />}{label}
      </div>
      <div className="mt-1 font-display text-3xl">{value}</div>
      {sub && <div className="text-sm text-muted-foreground">{sub}</div>}
    </div>
  );
}
function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="h-full rounded-xl border bg-card p-5">
      <h3 className="mb-3 text-lg font-semibold">{title}</h3>
      {children}
    </div>
  );
}
function FilePick({ accept, onFile, icon, label, disabled }: { accept: string; onFile: (f: File) => void; icon: React.ReactNode; label: string; disabled?: boolean }) {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <>
      <input ref={ref} type="file" accept={accept} className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      <Button variant="outline" disabled={disabled} onClick={() => ref.current?.click()}>{icon}{label}</Button>
    </>
  );
}
