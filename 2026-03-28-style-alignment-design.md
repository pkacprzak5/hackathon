# Style Alignment: Frontend to Pencil Design

**Date:** 2026-03-28
**Goal:** Align the frontend styling (colors, buttons, shadows, fonts, tab bar, cards) to match the Pencil design (`pencil-new.pen`) without changing content/structure affected by in-flight API integration PRs.

## Constraints

- **Style-only changes** — do not alter data flow, API hooks, component props that carry data, or content structure
- **CRITICAL: The codebase is actively being modified by other contributors (ongoing API integration work). Before making any change to a file, you MUST re-read it first to pick up the latest state. Never assume a file's contents match what was read earlier — always work against what is currently on disk.**
- **CSS variable changes cascade** — in-flight PRs will inherit the new look (accepted trade-off)
- Fonts switch to Plus Jakarta Sans (headings) + Inter (body)
- Tab bar gets a full visual redesign (pill style)
- Subtle liquid glass on 1-2 elements

---

## Layer 1: Foundation

### 1a. CSS Variable Changes (`globals.css`)

#### Light mode (`:root`)

| Variable | Current | New | Source |
|----------|---------|-----|--------|
| `--bg-card` | `#FFFFFF` | `#F4F4F5` | Pencil `$bg-card` |
| `--bg-elevated` | `#FAFAFA` | `#E4E4E7` | Pencil `$bg-elevated` |
| `--text-primary` | `#09090B` | `#18181B` | Pencil `$text-primary` |
| `--border` | `#E4E4E7` | `#D4D4D8` | Pencil `$border-std` |
| `--success` | `#22C55E` | `#14B8A6` | Pencil `$success` (teal) |

#### New variable

- `--purple-soft: #8B5CF620` — used for soft purple backgrounds (e.g., "You" leaderboard row, icon containers)

#### Body background

Change `body { background: var(--bg-surface) }` to `body { background: var(--bg-primary) }`.

Pages that explicitly set `bg-bg-surface` as their root background should switch to `bg-bg-primary`:
- `app/(player)/history/page.tsx`
- `app/(player)/profile/page.tsx`
- `app/(player)/results/page.tsx`

Note: `--success-muted` (`#14B8A6`) is now the same value as the updated `--success`. Keep both variables for compatibility — in-flight PRs may reference either.

#### Dark mode

Already well-aligned. No changes needed beyond inheriting the new variable names.

### 1b. Font Swap (following `next/font` best practices)

Use `next/font/google` for **zero-layout-shift**, self-hosted font loading. Never use `<link>` tags or `@import url()` for Google Fonts.

**In `app/layout.tsx`:**

```tsx
import { Inter, Plus_Jakarta_Sans } from 'next/font/google'

const inter = Inter({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-body',
  display: 'swap',        // Show fallback immediately, swap when ready
})

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ['latin'],
  weight: ['700', '800'],
  variable: '--font-heading',
  display: 'swap',
})
```

- Apply both font variables on `<html>`: `className={`${inter.variable} ${plusJakarta.variable}`}`
- Remove Geist font imports (no longer needed)
- Register in `@theme inline` block in `globals.css`:
  ```css
  --font-sans: var(--font-body);
  --font-heading: var(--font-heading);
  --color-purple-soft: var(--purple-soft);
  ```
- Set body font: `font-family: var(--font-body), system-ui, sans-serif`
- Headings/scores use `font-family: var(--font-heading)` explicitly via Tailwind class `font-heading`

**Important:** Fonts are loaded once in the root layout — never import `next/font/google` in individual components (creates duplicate instances).

### 1c. Bundle Optimization (`next.config.ts`)

Add `optimizePackageImports` to avoid barrel import penalty from `lucide-react`:

```ts
const nextConfig: NextConfig = {
  experimental: {
    optimizePackageImports: ['lucide-react'],
  },
};
```

### 1d. Card Border Removal

Remove `ring-1 ring-border-light` from all card elements across all pages. Pencil cards have no borders — they rely on gray-on-white contrast.

---

## Layer 2: Shared Components

### Tab Bar (`app/(player)/layout.tsx`)

Replace the current flat bottom bar with a Pencil-matching pill tab bar:

**Structure:**
```
Outer wrapper: padding 12px top, 21px sides/bottom
  Pill container: rounded-[36px], bg --bg-card, 1px border --border, h-[62px], p-1, flex horizontal
    Tab (active): rounded-[26px], gradient bg, flex-1, center, vertical, gap-1
      Icon: 18x18, white
      Label: 10px, weight 600, uppercase, letter-spacing 0.5px, white
    Tab (inactive): rounded-[26px], no bg, flex-1, center, vertical, gap-1
      Icon: 18x18, --text-muted
      Label: 10px, weight 500, uppercase, letter-spacing 0.5px, --text-muted
```

**Icons:** house, video, chart-bar, user (lucide)

### Button Patterns

| Variant | Height | Radius | Fill | Text | Example |
|---------|--------|--------|------|------|---------|
| Primary | 52px | 26px | Gradient (purple→pink) | White, 16px, 600 | End Session |
| Secondary | 48px | 24px | None, 2px gradient stroke | Purple, 14px, 600 | Export Data |
| Tertiary | 44px | 22px | None, 1.5px #D4D4D8 stroke | Dark, 13px, 600 | Link, Save |
| Small pill | auto | 100px | Gradient (active) / gray (inactive) | White/gray, 12px | Filter pills |

### Card Base Pattern

- Background: `bg-bg-card` (now `#F4F4F5`)
- Border-radius: `rounded-3xl` (24px)
- Padding: `p-5` (20px)
- No borders/rings
- Gap between cards: 12px

---

## Layer 3: Per-Page Adjustments

### Home (`app/(player)/home/page.tsx`)

**Remove:** Gradient banner header, overlapping card (-mt-10) pattern.

**Add:**
- Clean header row: "GymAI" gradient text (Plus Jakarta Sans, 28px, 800) + bell icon (24px, --text-secondary), space-between, px-6
- Score section: centered, vertical
  - Ring: 160px, 8px gradient stroke, #F4F4F5 inner fill
  - Score number: Plus Jakarta Sans, 48px, 800, --text-primary
  - "Average Form Score": Inter, 14px, 500, --text-secondary
- Stats row: two cards side-by-side, gap-4
  - Each: gray bg, rounded-3xl, p-5, vertical, gap-2
  - Gradient lucide icon (24px), number (Plus Jakarta Sans, 28px, 800), label (Inter, 13px, 500)
- Exercise cards: gray bg, rounded-3xl, p-5, horizontal, gap-4
  - 48px icon container: gradient bg (12% opacity), rounded-[14px]
  - Text column: name (bold) + subtitle (muted)
  - Chevron-right icon (20px, --text-muted)
- Multiplayer card: full gradient bg, white text/icons

### Leaderboard (`app/(player)/leaderboard/page.tsx`)

**Remove:** Gradient banner header.

**Add:**
- Plain "Leaderboard" heading (Plus Jakarta Sans, 28px, 800, --text-primary)
- Filter pills row: gap-2, horizontal
  - Active: gradient bg, rounded-full, py-2 px-4, white text 12px/600
  - Inactive: gray bg, rounded-full, py-2 px-4, --text-secondary 12px/500
  - "Today" pill: gray bg + 1px --border stroke
- Ranking rows: gap-2, vertical
  - Each row: bg-bg-card, rounded-2xl (16px), py-3 px-4, horizontal, gap-3
  - Rank number: Plus Jakarta Sans 16px/700 (gradient for #1, --text-secondary for others)
  - Avatar circle: 36px
  - Name + subtitle column
  - Score: Plus Jakarta Sans 20px/800
  - "You" row: bg `--purple-soft`, 1.5px purple border, purple score text
- "Show top 50" link: centered, purple text 13px/600

### Profile (`app/(player)/profile/page.tsx`)

**Changes:**
- "Profile" heading: Plus Jakarta Sans, 28px, 800
- Avatar: change from rounded square to circle with 3px gradient stroke
- Settings sections: grouped cards with section header (14px/600) inside card
  - Rows: justify-between, label (14px, normal) + value/toggle
  - Toggle switches: gradient bg when active, gray (#E4E4E7) when off
- Export button: secondary outlined style (48px, gradient 2px stroke, purple text)

### History (`app/(player)/history/page.tsx`)

**Changes:**
- "History" heading: Plus Jakarta Sans, 28px, 800
- Cards: gray bg, no ring borders, rounded-3xl
- Score highlights: gradient/teal coloring

### Results (`app/(player)/results/page.tsx`)

**Remove:** Gradient banner header.

**Add:**
- Trophy icon (40px) with gradient fill, centered
- "Alex Wins!" gradient text (Plus Jakarta Sans, 28px, 800), centered
- Side-by-side result cards:
  - Winner: bg #8B5CF610, 2px gradient stroke, rounded-[20px]
  - Loser: gray bg, rounded-[20px]
  - Each: avatar, name, large score, "X correct reps"
- Chart card: gray bg, rounded-[20px], "Knee Angle Comparison"
- Video thumbnail: dark bg (#1A1A2E), rounded-2xl, play button overlay
- Button row: Share (primary gradient, 44px) / Link (tertiary) / Save (tertiary), gap-[10px]

### Session (`app/(player)/session/*/page.tsx`)

**Changes (style only, no camera/API changes):**
- Joint angle bars: teal (#14B8A6) for good, red (#EF4444) for bad, gray track (#F4F4F5 light / #FFFFFF15 dark)
- AI feedback bubble: rounded-[18px], gray bg, sparkles icon with gradient, p-3/p-4
- End Session button: primary gradient (52px, rounded-[26px])

### Replay (`app/(player)/replay/*/page.tsx`)

**Changes:**
- Match session styling tokens
- Stat pills at bottom: gray bg, rounded-xl

---

## Layer 4: Polish

### Liquid Glass (2 spots)

1. **Tab bar pill**: `backdrop-blur-xl bg-white/70 dark:bg-[#1A1A2E]/70` instead of solid fill. Keep border.
2. **AI feedback bubble** (session pages): `backdrop-blur-md bg-bg-card/80` + 1px translucent border

### Fine-tuning

- Page content padding: 24px (px-6) to match Pencil's 24px padding
- Card gaps: 12px between exercise cards, 8px between leaderboard rows, 16px between stat cards
- Gradient direction: Pencil uses rotation 90 = left-to-right for horizontal gradients. Verify `bg-linear-to-r` matches.
- Dark mode pass: verify all changes look correct in dark mode with existing dark tokens
- Score ring gradient: verify stroke uses gradient (purple→pink) not solid color

---

## Files Affected

### Layer 1 (Foundation)
- `app/globals.css` — CSS variables, body background
- `app/layout.tsx` — Font loading
- All pages with `ring-1 ring-border-light` — Remove card borders

### Layer 2 (Shared Components)
- `app/(player)/layout.tsx` — Tab bar redesign

### Layer 3 (Per-page)
- `app/(player)/home/page.tsx`
- `app/(player)/leaderboard/page.tsx`
- `app/(player)/profile/page.tsx`
- `app/(player)/history/page.tsx`
- `app/(player)/results/page.tsx`
- `app/(player)/session/solo/page.tsx`
- `app/(player)/session/multi/page.tsx`
- `app/(player)/replay/*/page.tsx`

### Layer 4 (Polish)
- `app/(player)/layout.tsx` — Glass effect on tab bar
- Session components — Glass effect on AI feedback

---

## Next.js & React Best Practices to Follow

### Font Loading (`next/font`)
- Use `next/font/google` exclusively — self-hosts fonts, eliminates network requests, prevents layout shift (CLS = 0)
- Specify `display: 'swap'` for instant fallback rendering
- Only load needed weights (700, 800 for headings; 400, 500, 600 for body) — avoid loading all weights
- Always specify `subsets: ['latin']` to minimize font file size
- Export font instances from root layout only, never from individual components

### Bundle Optimization (`next.config.ts`)
- Add `optimizePackageImports: ['lucide-react']` — the app imports many lucide icons across pages. Without this, barrel imports pull in all 1,500+ icons (~2.8s dev penalty, 200-800ms cold start). This transforms imports to direct paths at build time with zero TypeScript loss.
- This is especially important since the tab bar redesign and per-page changes add new lucide icon imports.

### RSC Boundary Awareness
- The tab bar in `app/(player)/layout.tsx` uses `'use client'` (it calls `usePathname()`). Style changes stay within this client component — no RSC boundary issues.
- Page components that are Server Components (most pages) should keep styling in Tailwind classes and CSS variables. Do NOT introduce `useState`/`useEffect` for styling purposes.
- If a page needs client-side theme detection beyond what `next-themes` provides, use the existing `ThemeProvider` — don't add new client boundaries.

### Hydration Safety
- The app already uses `next-themes` with `suppressHydrationWarning` on `<html>` — this correctly prevents hydration mismatch for dark mode class switching.
- The existing inline `<script>` pattern for service worker registration is correct — it runs after hydration.
- When adding glass/translucent effects that depend on theme, use CSS variables (already theme-aware) rather than JS-based theme detection to avoid hydration flicker.

### Performance Patterns
- **No inline component definitions:** When restructuring the tab bar or cards, define subcomponents (e.g., `TabItem`) at module level, not inside the parent component. Inline definitions cause remount on every render, losing state and restarting animations.
- **`content-visibility: auto`** — Consider adding to leaderboard ranking rows and history session lists. This lets the browser skip layout/paint for off-screen items, improving initial render for long lists. Add via CSS:
  ```css
  .leaderboard-row { content-visibility: auto; contain-intrinsic-size: 0 64px; }
  ```
- **Prefer CSS over JS for visual changes:** All style alignment should use Tailwind classes and CSS variables. Avoid adding `useEffect` hooks for style calculations — use CSS `calc()`, variables, and media queries instead.
- **Gradient text technique:** Use `bg-clip-text text-transparent bg-linear-to-r` rather than SVG or canvas for gradient text. This is pure CSS, no JS, and works with SSR.

### Dark Mode Implementation
- CSS variables already cascade correctly between `:root` and `.dark` — the existing architecture handles this well.
- For new glass effects, use CSS variables with alpha channels (e.g., `--bg-card` with `/70` opacity modifier in Tailwind) so both themes work from a single class definition.
- Test both themes after each layer of changes — don't defer all dark mode testing to Layer 4.

---

## What Is NOT Changing

- Component props, data flow, API hooks, WebSocket logic
- Coach dashboard layout/structure
- Camera feed components (structure)
- Any TypeScript interfaces or types
- Route structure
- Provider logic
