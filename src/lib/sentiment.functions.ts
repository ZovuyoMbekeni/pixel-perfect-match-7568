import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

export const SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest";
export const VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct";
const INSIGHTS_MODEL = "openai/gpt-6-astra";

export type SentimentLabel = "Positive" | "Negative" | "Neutral";
export interface Prediction {
  label: SentimentLabel;
  confidence: number;
  scores: Record<SentimentLabel, number>;
}

function hfKey() {
  const key = process.env["HUGGINGFACE_API_KEY"];
  if (!key) throw new Error("The Hugging Face API key is not configured on the server (HUGGINGFACE_API_KEY).");
  return key;
}

function normalise(label: string): SentimentLabel {
  const l = label.toLowerCase();
  if (l.includes("pos") || l === "label_2") return "Positive";
  if (l.includes("neg") || l === "label_0") return "Negative";
  return "Neutral";
}

async function classifyBatch(texts: string[], key: string): Promise<Prediction[]> {
  const res = await fetch(
    `https://router.huggingface.co/hf-inference/models/${SENTIMENT_MODEL}`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: texts, options: { wait_for_model: true }, parameters: { top_k: 3, truncation: true } }),
    },
  );
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Hugging Face sentiment service error (${res.status}): ${body.slice(0, 200)}`);
  }
  const data = (await res.json()) as unknown;
  if (!Array.isArray(data)) throw new Error("Unexpected response from the sentiment model.");
  // single input may come back flat
  const rows: { label: string; score: number }[][] =
    texts.length === 1 && data.length && !Array.isArray(data[0]) ? [data as never] : (data as never);
  if (rows.length !== texts.length) throw new Error("The sentiment model returned an incomplete result.");
  return rows.map((row) => {
    const scores: Record<SentimentLabel, number> = { Positive: 0, Negative: 0, Neutral: 0 };
    for (const r of row) scores[normalise(r.label)] = r.score;
    const top = [...row].sort((a, b) => b.score - a.score)[0];
    return { label: normalise(top.label), confidence: top.score, scores };
  });
}

export const analyzeTexts = createServerFn({ method: "POST" })
  .inputValidator((d) => z.object({ texts: z.array(z.string().min(1).max(2000)).min(1).max(500) }).parse(d))
  .handler(async ({ data }) => {
    const key = hfKey();
    const out: Prediction[] = [];
    for (let i = 0; i < data.texts.length; i += 16) {
      out.push(...(await classifyBatch(data.texts.slice(i, i + 16), key)));
    }
    return { model: SENTIMENT_MODEL, predictions: out };
  });

export const extractImageText = createServerFn({ method: "POST" })
  .inputValidator((d) =>
    z.object({ dataUrl: z.string().regex(/^data:image\/(png|jpe?g);base64,/).max(6_000_000) }).parse(d),
  )
  .handler(async ({ data }) => {
    const key = hfKey();
    const res = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: VISION_MODEL,
        max_tokens: 1500,
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text:
                  "Transcribe every comment, review or piece of feedback text visible in this image exactly as written. Output one comment per line, no numbering, no commentary. Skip usernames, timestamps, like counts and UI labels. If there is no readable text, output exactly: NO_TEXT_FOUND",
              },
              { type: "image_url", image_url: { url: data.dataUrl } },
            ],
          },
        ],
      }),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Hugging Face image service error (${res.status}): ${body.slice(0, 200)}`);
    }
    const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
    const text = json.choices?.[0]?.message?.content?.trim() ?? "";
    if (!text || text.includes("NO_TEXT_FOUND")) return { model: VISION_MODEL, lines: [] as string[] };
    const lines = text
      .split(/\n+/)
      .map((l) => l.replace(/^[-*•\d.)\s"]+|"$/g, "").trim())
      .filter((l) => l.length > 1);
    return { model: VISION_MODEL, lines };
  });

export const generateInsights = createServerFn({ method: "POST" })
  .inputValidator((d) =>
    z
      .object({
        stats: z.object({ total: z.number(), positive: z.number(), negative: z.number(), neutral: z.number(), avgConfidence: z.number() }),
        items: z.array(z.object({ text: z.string(), label: z.string(), confidence: z.number() })).max(300),
      })
      .parse(d),
  )
  .handler(async ({ data }) => {
    const apiKey = process.env["LOVABLE_API_KEY"];
    if (!apiKey) throw new Error("The insights service is not configured.");
    const s = data.stats;
    const pct = (n: number) => (s.total ? Math.round((n / s.total) * 100) : 0);
    const prompt = `You are a data analyst. Below are comments that a Hugging Face sentiment model (${SENTIMENT_MODEL}) has already classified. Do NOT reclassify them.
Statistics: total ${s.total}; positive ${s.positive} (${pct(s.positive)}%); negative ${s.negative} (${pct(s.negative)}%); neutral ${s.neutral} (${pct(s.neutral)}%); average confidence ${(s.avgConfidence * 100).toFixed(1)}%.

Comments:
${data.items.map((i) => `[${i.label} ${(i.confidence * 100).toFixed(0)}%] ${i.text.slice(0, 300)}`).join("\n")}

Write 4 to 6 concise insights grounded only in this data: overall sentiment, recurring themes behind negative and positive comments (quote short phrases), notable low-confidence predictions, and one practical recommendation. Return plain text, one insight per line starting with "- ". No headings.`;

    const res = await fetch("https://ai.gateway.lovable.dev/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Lovable-API-Key": apiKey, "X-Lovable-AIG-SDK": "fetch" },
      body: JSON.stringify({ model: INSIGHTS_MODEL, input: prompt, stream: true, store: false, reasoning: { effort: "low" } }),
    });
    if (!res.ok || !res.body) {
      if (res.status === 429) throw new Error("The insights service is busy. Please try again in a moment.");
      if (res.status === 402) throw new Error("AI credits are used up for this workspace. Add credits to generate insights.");
      throw new Error(`Insights service error (${res.status}).`);
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
        try {
          const ev = JSON.parse(payload) as { type?: string; delta?: string; error?: { message?: string } };
          if (ev.type === "response.output_text.delta" && ev.delta) text += ev.delta;
          if (ev.type === "error" || ev.type === "response.failed") throw new Error(ev.error?.message ?? "Insights generation failed.");
        } catch (e) {
          if (e instanceof SyntaxError) continue;
          throw e;
        }
      }
    }
    const insights = text.split("\n").map((l) => l.replace(/^[-*•]\s*/, "").trim()).filter(Boolean);
    if (!insights.length) throw new Error("The insights service returned no content.");
    return { model: INSIGHTS_MODEL, insights };
  });
