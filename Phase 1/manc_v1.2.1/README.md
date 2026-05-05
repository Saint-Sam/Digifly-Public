# MANC v1.2.1 Local SWC Cache

This folder is the local public-cache location for MANC v1.2.1 SWC exports used by Phase 2 launchers.

Runtime data is stored under:

```text
export_swc/
```

`export_swc/` is intentionally ignored by git because SWC exports and sidecar files can be large. The VIP glia mutation launchers can seed this cache by copying requested neuron folders from a configured local source export such as:

```text
/path/to/source/export_swc
```

Copied folders preserve the Phase 1 export convention, for example:

```text
export_swc/DN/DNp01/10000/
export_swc/MN/TTMn/10068/
export_swc/IN/PSI/11446/
```
