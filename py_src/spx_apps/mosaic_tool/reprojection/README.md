# Reprojection Methods

The reprojection algorithms in this directory were adopted from the SPHEREx Map Maker.

## Astropy's reproject_exact

`reproject_exact` from the Astropy `reproject` package uses a **flux-conserving spherical polygon intersection** algorithm.

### Algorithm

1. **Treat each pixel as a spherical polygon**: Each input pixel is mapped to a polygon on the celestial sphere using its WCS.

2. **Calculate exact overlap areas**: For each input pixel, compute the exact geometric intersection area with every output pixel it overlaps.

3. **Distribute flux by area fraction**: The input pixel's value is distributed to output pixels proportionally to the overlap area:

   ```
   output_pixel += input_pixel_value × (overlap_area / input_pixel_area)
   ```

4. **Sum contributions**: Each output pixel accumulates weighted contributions from all overlapping input pixels.


### Diagram

```
Input Grid                      Output Grid
┌───┬───┬───┐                  ┌─────┬─────┐
│ A │ B │ C │                  │     │     │
├───┼───┼───┤    ──────►       │  X  │  Y  │
│ D │ E │ F │   reproject      │     │     │
├───┼───┼───┤    exact         ├─────┼─────┤
│ G │ H │ I │                  │  Z  │  W  │
└───┴───┴───┘                  └─────┴─────┘

Example: Output pixel X overlaps with input pixels A, B, D, E
  X = A×(overlap_A/area_A) + B×(overlap_B/area_B) +
      D×(overlap_D/area_D) + E×(overlap_E/area_E)
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Flux conservation** | Total flux is preserved (no flux lost or created) |
| **Fractional hits** | nhits can be non-integer (fractional pixel overlaps) |
| **Handles distortion** | Properly accounts for WCS distortions and varying pixel scales |
| **Slow** | O(n×m) comparisons for n input × m output pixels |


## 2. Histogram-Based Reprojection

`reproject_hist` is a **histogram-based (pointillist) reprojection** method from the SPHEREx Map Maker (not Astropy).

### Algorithm

1. **Get input pixel centers**: For each input pixel, compute its (RA, Dec) sky coordinate from WCS.

2. **Map to output grid**: Convert each sky coordinate to the nearest output pixel using the output WCS.

3. **Bin with histogram**: Use 2D histogram to accumulate values into output pixels.

```
# Core logic (simplified from reproject_tools.py:7-62)
raExp, decExp = wcsExp.all_pix2world(ixExp, iyExp, 0)    # Input pixel → sky
ixMos, iyMos = wcsMos.all_world2pix(raExp, decExp, 0)    # Sky → output pixel

# Histogram binning
nhits = histogram2d(ixMos, iyMos)                        # Count hits
acc   = histogram2d(ixMos, iyMos, weights=data)          # Sum values
mosaic = acc / nhits                                     # Average
```
### Step-by-Step Example

```
Input pixel at (100, 200) with value = 5.0 MJy/sr
    ↓
WCS transform: (100, 200) → (RA=270.5°, Dec=66.3°)
    ↓
Inverse WCS: (RA, Dec) → output pixel (45, 72)
    ↓
Histogram bin (45, 72):
    acc[45, 72] += 5.0
    nhits[45, 72] += 1
    ↓
After all pixels: mosaic[45, 72] = acc[45, 72] / nhits[45, 72]
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Fast** | Simple histogram, no geometric calculations |
| **Integer hits** | nhits is always integer (count of input pixels) |
| **Gaps possible** | Single exposure may not fill all output pixels |
| **No smoothing** | Each input pixel goes to exactly one output pixel |
| **Preserves noise** | No interpolation = no noise correlation |
| **Nearest-neighbor** | Input pixel assigned to closest output pixel center |

### Comparison with Astropy's Nearest-Neighbor

Astropy's `reproject_interp` has a **nearest-neighbor** option, but it works in the **opposite direction**:

| Method | Direction | Logic |
|--------|-----------|-------|
| `reproject_hist` | Input → Output | For each **input** pixel, find nearest **output** pixel, accumulate |
| `reproject_interp(order='nearest-neighbor')` | Output → Input | For each **output** pixel, find nearest **input** pixel, sample |

### Diagram

```
reproject_hist (input → output):
┌─────────────────┐         ┌─────────────┐
│ Input pixels    │         │ Output grid │
│  A  B  C  D     │ ──────► │    [X]      │  Multiple inputs → same output
│  (all land in X)│         │             │  X = average(A,B,C,D)
└─────────────────┘         └─────────────┘

reproject_interp nearest (output → input):
┌─────────────────┐         ┌─────────────┐
│ Input pixels    │         │ Output grid │
│  A  B  C  D     │ ◄────── │    [X]      │  Output samples nearest input
│     ↑           │         │             │  X = B (nearest to X's center)
│   nearest       │         │             │
└─────────────────┘         └─────────────┘
```

### Key Differences

| Aspect | `reproject_hist` | `reproject_interp (nearest)`                |
|--------|------------------|------------------------------------|
| Multiple inputs → 1 output | Averaged | Only nearest sampled               |
| Flux conservation | Yes (approximately) | No                                 |
| Information loss | None | Loses data from non-sampled inputs |
| Gaps in output | Possible | No (always samples something)      |
| nhits meaningful | Yes | Not really                         |
