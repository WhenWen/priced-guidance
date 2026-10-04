# Threat model

Generator and oracle code, dependencies, archives, messages, logs, and idea text are untrusted. The referee, pricing and branch code, judge, model gateway, result store, and evaluation profile are trusted.

The local runner separates roles into JSON-speaking processes and withholds target constructor data from the generator. It is for public and development targets only: those processes still share a host kernel and can access ambient files and networks, so their scores are unverified.

Hidden targets require an independently reviewed external worker with separate hardware-backed guests (or an equivalently reviewed userspace-kernel profile), no shared filesystems or network, arena-owned credentials, externally enforced limits, erased per-run state, and attested profile versions. This repository fails closed when `--runner hardened` is requested without such a worker.

In scope:

- target exfiltration and cross-role collusion;
- malformed messages, archives, probabilities, and checkout handles;
- nondeterminism, replay divergence, and rewinding resource counters;
- inference of intermediate Judge rejection from public progress events;
- prompt injection against the judge;
- denial of service within declared resource limits.

Not claimed solved:

- compromise of the trusted referee or hardened worker;
- kernel, hypervisor, firmware, physical, or sophisticated hardware side channels;
- correctness of one LLM judge against all semantic attacks.

Before hidden evaluation, review and test the concrete worker, broker, kernel, model gateway, disclosure policy, judge, and incident response—not only this protocol code.
