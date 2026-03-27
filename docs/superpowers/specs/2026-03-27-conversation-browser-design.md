## Conversation Browser Modal

Two-panel modal for browsing session exchanges with hierarchical folding, ontology labels, and token weight coloring.

### Layout
- 90% screen width/height modal, `e` keybinding from session browser
- Left panel (75%): Tree with collapsible section nodes
- Right panel (25%): Rendered conversation preview of selected exchange

### Tree Structure
- Fold N exchanges into 100 sections, recurse until ≤50 at leaf
- Section nodes: `▸ §1-100 [████░░] 42k tok  "last KEEP or user text"`
- Leaf exchanges: `261 user: KEEP "deploy to prod"`
- Token weight bar: 6-char proportional to section's share of total
- Color gradient: green (bottom 25%), amber (25-75%), red (top 25%)

### Ontology Labels
- KEEP: ★ prefix, blue highlight
- DISTILL: dim/muted style
- HEURISTIC: default
- Untagged: no special treatment

### Right Panel
Rendered conversation: role-colored (green=user, blue=assistant), word-wrapped, scrollable.

### Keybindings
- `e`: open from session browser
- `enter`/`space`: toggle fold
- `j`/`k`/arrows: navigate
- `escape`: close
- Future: `d` to delete message

### Data Loading
Worker thread parses full JSONL, builds exchange list with _reduce tags and token sizes. Tree built from exchange list on mount.
