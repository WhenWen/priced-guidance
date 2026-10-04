# Output-token defaults for future runs

From September 5, 2026, new structured-model requests default to **50,000
output tokens**. The service gateway's default per-call limit is also 50,000,
so it does not silently reduce a reference request to the former 32,000 cap.
The reference Generator and Oracle use this allowance in every stage and
channel, and the research Judge uses it too. Direct callers can still request
a smaller allowance; an explicitly configured service limit remains binding.

This is an allowance, not a required output length or a dollar budget. In the
OpenAI Responses API it includes **reasoning plus visible output**. A response
can exhaust the allowance before producing any JSON. More headroom reduces
that risk but cannot guarantee completion. See the
[official Responses reference](https://developers.openai.com/api/reference/python/resources/responses/methods/retrieve).

OpenAI parse retries still grow smaller allowances by 1.5x, but stop growing
at 50,000; default requests do not silently expand to 75,000 or 112,500. A
direct provider caller explicitly requesting more than 50,000 keeps that
explicit allowance. Existing non-OpenAI provider-specific retry policies are
unchanged. Request timeouts, actor deadlines, retry counts, monetary budgets,
and model/reasoning selection are unchanged.

The new reference submission versions are `reference-pair` **1.17.4** and
`reference-pair-fable51` **1.17.5**. Their request caps are part of their frozen
submission hashes. This is a new-run policy, not a compatible edit to an
already-frozen trajectory or a retroactive repair of historical results.
