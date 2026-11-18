[API reference](./index.html) › **Decision Parsing**

---

# Decision Parsing

Validate and parse unstructured data by asking a decision model, and judge
existing columns with semantic checks. See the [decision parsing guide](../decisions.html) and the [tutorial](../decision_parsing_tutorial.html) for how to use these objects.

## Question Types

A decision model answers typed questions; the question type says what shape
the answer has. These are the parser objects passed to `ParsedField` /
`ParsedColumn`.

- [`Choice`](./generated/pandera.decisions.Choice.html)
- [`Score`](./generated/pandera.decisions.Score.html)
- [`Noul`](./generated/pandera.decisions.Noul.html)
- [`Confidence`](./generated/pandera.decisions.Confidence.html)

## Providers

- [`DecisionProvider`](./generated/pandera.decisions.DecisionProvider.html)
- [`MockProvider`](./generated/pandera.decisions.MockProvider.html)
- [`RecordingProvider`](./generated/pandera.decisions.RecordingProvider.html)
- [`ReplayProvider`](./generated/pandera.decisions.ReplayProvider.html)

## Configuration

- [`set_provider`](./generated/pandera.decisions.set_provider.html)
- [`get_provider`](./generated/pandera.decisions.get_provider.html)
- [`provider`](./generated/pandera.decisions.provider.html)
- [`register_provider`](./generated/pandera.decisions.register_provider.html)
- [`capabilities_of`](./generated/pandera.decisions.capabilities_of.html)
- [`enabled`](./generated/pandera.decisions.enabled.html)

## Semantic Checks

- [`Holds`](./generated/pandera.decisions.Holds.html)
- [`holds`](./generated/pandera.decisions.holds.html)

## Caching and Inspection

- [`MemoryCache`](./generated/pandera.decisions.MemoryCache.html)
- [`SQLiteCache`](./generated/pandera.decisions.SQLiteCache.html)
- [`Plan`](./generated/pandera.decisions.Plan.html)

- [`questions`](./generated/pandera.decisions.questions.html)
- [`plan`](./generated/pandera.decisions.plan.html)
- [`stats`](./generated/pandera.decisions.stats.html)

## Types and Errors

- [`Decision`](./generated/pandera.decisions.Decision.html)
- [`ProviderCapabilities`](./generated/pandera.decisions.ProviderCapabilities.html)
- [`ProviderLimits`](./generated/pandera.decisions.ProviderLimits.html)
- [`DecisionsConfigError`](./generated/pandera.decisions.DecisionsConfigError.html)
- [`ProviderCapabilityError`](./generated/pandera.decisions.ProviderCapabilityError.html)

`Question` is a type alias (a union of the three question kinds) rather than
an instantiable class; it appears in the signatures of the objects above.
