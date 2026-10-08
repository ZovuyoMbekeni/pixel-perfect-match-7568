<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

## Architecture
- All AI calls live in `src/lib/sentiment.functions.ts` server functions — keeps LOVABLE_API_KEY off the client.
- Classification and insights are separate gateway calls; insights never reclassify — keeps labels and narration independent. No third-party keys required.
- Analysis results stay in browser state, no database — privacy requirement (nothing stored).
