import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

export const SENTIMENT_MODEL = "openai/gpt-6-astra";
export const VISION_MODEL = "openai/gpt-6-astra";
const INSIGHTS_MODEL = "openai/gpt-6-astra";

export type SentimentLabel = "Positive" | "Negative" | "Neutral";
export interface Prediction {
  label: SentimentLabel;
  confidence: number;
  scores: Record<SentimentLabel, number>;
}

async function callGateway(input: unknown, format?: Record<string, unknown>): Promise<string> {
  const apiKey = process.env["LOVABLE_API_KEY"];
  if (!apiKey) throw new Error("The analysis service is not configured.");
  const res = await fetch("https://ai.gateway.lovable.dev/v1/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Lovable-API-Key": apiKey, "X-Lovable-AIG-SDK": "fetch" },
    body: JSON.stringify({
      model: SENTIMENT_MODEL, input, stream: true, store: false, reasoning: { effort: "low" },
      ...(format ? { text: { format } } : {}),
    }),
  });
  if (!res.ok || !res.body) {
    if (res.status === 429) throw new Error("The analysis service is busy. Please try again in a moment.");
    if (res.status === 402) throw new Error("Credits are used up for this workspace. Add credits to keep analysing.");
    if (res.status === 403) throw new Error("Access to the analysis service is blocked for this workspace.");
    throw new Error(`Analysis service error (${res.status}).`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n");
    buf = parts.pop() ?? "";
    for (const line of parts) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;
      let ev: { type?: string; delta?: string; error?: { message?: string }; response?: { error?: { message?: string } } };
      try { ev = JSON.parse(payload); } catch { continue; }
      if (ev.type === "response.output_text.delta" && ev.delta) text += ev.delta;
      if (ev.type === "error" || ev.type === "response.failed")
        throw new Error(ev.error?.message ?? ev.response?.error?.message ?? "Analysis failed.");
    }
  }
  return text.trim();
}

const classifyFormat = {
  type: "json_schema",
  name: "sentiment",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    required: ["results"],
    properties: {
      results: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["index", "positive", "negative", "neutral"],
          properties: {
            index: { type: "integer" },
            positive: { type: "number" },
            negative: { type: "number" },
            neutral: { type: "number" },
          },
        },
      },
    },
  },
};

async function classifyBatch(texts: string[]): Promise<Prediction[]> {
  const prompt = `Classify the sentiment of each numbered comment. For each, give probabilities for positive, negative and neutral that sum to 1. Return one result per comment with its index.\n\n${texts
    .map((t, i) => `${i}: ${t.replace(/\s+/g, " ").slice(0, 1000)}`)
    .join("\n")}`;
  const raw = await callGateway(prompt, classifyFormat);
  const parsed = JSON.parse(raw) as { results: { index: number; positive: number; negative: number; neutral: number }[] };
  return texts.map((_, i) => {
    const r = parsed.results.find((x) => x.index === i);
    if (!r) throw new Error("The sentiment service returned an incomplete result.");
    const sum = r.positive + r.negative + r.neutral || 1;
    const scores: Record<SentimentLabel, number> = { Positive: r.positive / sum, Negative: r.negative / sum, Neutral: r.neutral / sum };
    const [label, confidence] = (Object.entries(scores) as [SentimentLabel, number][]).sort((a, b) => b[1] - a[1])[0]!;
    return { label, confidence, scores };
  });
}

export const analyzeTexts = createServerFn({ method: "POST" })
  .inputValidator((d) => z.object({ texts: z.array(z.string().min(1).max(2000)).min(1).max(500) }).parse(d))
  .handler(async ({ data }) => {
    const out: Prediction[] = [];
    for (let i = 0; i < data.texts.length; i += 40) out.push(...(await classifyBatch(data.texts.slice(i, i + 40))));
    return { model: SENTIMENT_MODEL, predictions: out };
  });

export const extractImageText = createServerFn({ method: "POST" })
  .inputValidator((d) =>
    z.object({ dataUrl: z.string().regex(/^data:image\/(png|jpe?g);base64,/).max(6_000_000) }).parse(d),
  )
  .handler(async ({ data }) => {
    const text = await callGateway([
      {
        role: "user",
        content: [
          { type: "input_text", text: "Transcribe every comment, review or piece of feedback text visible in this image exactly as written. Output one comment per line, no numbering, no commentary. Skip usernames, timestamps, like counts and UI labels. If there is no readable text, output exactly: NO_TEXT_FOUND" },
          { type: "input_image", image_url: data.dataUrl },
        ],
      },
    ]);
    if (!text || text.includes("NO_TEXT_FOUND")) return { model: VISION_MODEL, lines: [] as string[] };
    const lines = text.split(/\n+/).map((l) => l.replace(/^[-*•\d.)\s"]+|"$/g, "").trim()).filter((l) => l.length > 1);
    return { model: VISION_MODEL, lines };
  });

export const generateInsights = createServerFn({ method: "POST" })
  .inputValidator((d) =>
    z.object({
      stats: z.object({ total: z.number(), positive: z.number(), negative: z.number(), neutral: z.number(), avgConfidence: z.number() }),
      items: z.array(z.object({ text: z.string(), label: z.string(), confidence: z.number() })).max(300),
    }).parse(d),
  )
  .handler(async ({ data }) => {
    const s = data.stats;
    const pct = (n: number) => (s.total ? Math.round((n / s.total) * 100) : 0);
    const prompt = `You are a data analyst. Below are comments that have already been classified for sentiment. Do NOT reclassify them.
Statistics: total ${s.total}; positive ${s.positive} (${pct(s.positive)}%); negative ${s.negative} (${pct(s.negative)}%); neutral ${s.neutral} (${pct(s.neutral)}%); average confidence ${(s.avgConfidence * 100).toFixed(1)}%.

Comments:
${data.items.map((i) => `[${i.label} ${(i.confidence * 100).toFixed(0)}%] ${i.text.slice(0, 300)}`).join("\n")}

Write 4 to 6 concise insights grounded only in this data: overall sentiment, recurring themes behind negative and positive comments (quote short phrases), notable low-confidence predictions, and one practical recommendation. Return plain text, one insight per line starting with "- ". No headings.`;
    const text = await callGateway(prompt);
    const insights = text.split("\n").map((l) => l.replace(/^[-*•]\s*/, "").trim()).filter(Boolean);
    if (!insights.length) throw new Error("The insights service returned no content.");
    return { model: INSIGHTS_MODEL, insights };
  });
