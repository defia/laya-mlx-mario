"""Super Mario Bros driven by a local Laya model (laya-mlx).

Port of fhshaik/typesafe-mario: the NES emulator state parser, the seven
controller macros, the realtime dashboard, and the episode runner are kept;
the remote TypeSafe/Jev API policy is replaced by `LayaPolicy`, which answers
the same three typed judgments (choice / noul / score) with one local
`Agent.predict` call — no network, ~10ms per decision.
"""
