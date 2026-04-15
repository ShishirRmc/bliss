# Bliss UI Design System
> Extracted and adapted from the Cinematic Landing Page Builder prompt.
> Preset: **Midnight Luxe** (Dark Editorial)
> Identity: A private members' club meets a high-end wellness atelier.

---

## Palette

| Token       | Hex       | Usage                              |
|-------------|-----------|------------------------------------|
| Obsidian    | `#0D0D12` | App background, deepest surfaces   |
| Slate       | `#2A2A35` | Panels, cards, sidebar             |
| Champagne   | `#C9A84C` | Accent — CTAs, highlights, active  |
| Ivory       | `#FAF8F5` | Primary text                       |
| Muted       | `#6B6B7B` | Secondary text, placeholders       |
| Border      | `#1F1F2E` | Subtle dividers                    |

---

## Typography

- **Headings / UI labels:** `Inter` — tight tracking (`letter-spacing: -0.02em`)
- **Drama / empty states / large callouts:** `Playfair Display` Italic
- **Data / scores / IDs / monospace:** `JetBrains Mono`

Load via Google Fonts in `index.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:ital@1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

---

## Visual Texture

Apply a global noise overlay to eliminate flat digital gradients.
Add this once to `index.css` — it sits on top of everything at pointer-events: none:

```css
body::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 9999;
  opacity: 0.05;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23noise)' opacity='1'/%3E%3C/svg%3E");
}
```

---

## Radius System

No sharp corners anywhere.

| Context              | Radius         |
|----------------------|----------------|
| Cards / panels       | `rounded-3xl` (1.5rem) |
| Buttons              | `rounded-xl` (0.75rem) |
| Inputs               | `rounded-xl` (0.75rem) |
| Pills / badges       | `rounded-full` |
| Sidebar              | No radius (full height edge) |

---

## Micro-Interactions

### Magnetic Buttons
All buttons get a subtle scale on hover with a premium easing curve.
```css
.btn {
  transition: transform 200ms cubic-bezier(0.25, 0.46, 0.45, 0.94),
              box-shadow 200ms cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.btn:hover {
  transform: scale(1.03);
}
```

### Sliding Background (primary CTA buttons)
Buttons use `overflow-hidden` with an inner `<span>` that slides in on hover:
```jsx
<button className="relative overflow-hidden group ...">
  <span className="absolute inset-0 bg-champagne translate-y-full group-hover:translate-y-0
    transition-transform duration-300 ease-out" />
  <span className="relative z-10">Send</span>
</button>
```

### Link / Interactive Element Lift
```css
transition: transform 150ms ease;
&:hover { transform: translateY(-1px); }
```

---

## Animation (GSAP)

Install: `npm install gsap`

### Setup pattern
```js
import { gsap } from 'gsap'
import { useEffect, useRef } from 'react'

useEffect(() => {
  const ctx = gsap.context(() => {
    // animations here
  }, rootRef)
  return () => ctx.revert()
}, [])
```

### Entrance (fade-up)
```js
gsap.from(el, { y: 24, opacity: 0, duration: 0.6, ease: 'power3.out' })
```

### Stagger values
- Text elements: `stagger: 0.08`
- Cards / containers: `stagger: 0.15`

### Morph / state transitions
```js
ease: 'power2.inOut'
```

---

## Navbar

Fixed, pill-shaped, horizontally centered.

**Morphing logic:**
- Default: transparent background, light text
- On scroll past ~80px: `bg-[#0D0D12]/70 backdrop-blur-xl` + champagne border
- Use `IntersectionObserver` or a scroll listener

```jsx
// scroll listener pattern
useEffect(() => {
  const onScroll = () => setScrolled(window.scrollY > 80)
  window.addEventListener('scroll', onScroll)
  return () => window.removeEventListener('scroll', onScroll)
}, [])
```

---

## Cards — "Functional Software Micro-UI" Philosophy

Cards must feel like live instruments, not static containers.
- Background: `#2A2A35` (Slate)
- Border: `1px solid #1F1F2E`
- Radius: `rounded-3xl`
- Shadow: `shadow-[0_8px_32px_rgba(0,0,0,0.4)]`
- On hover: subtle `translateY(-2px)` lift + shadow deepens

### RecommendationCard specifics
- Confidence score: animated progress bar (CSS transition width on mount via GSAP or useEffect)
- Category badge: pill with champagne tint for high-confidence, muted for low
- Price: `JetBrains Mono` font
- Rank number: `JetBrains Mono`, muted, small

### MessageBubble specifics
- User bubble: champagne-tinted background (`#C9A84C1A` + champagne border)
- Assistant bubble: slate background
- Source badges: pill shape, `JetBrains Mono`, champagne text
- Entrance: GSAP fade-up (`y: 12 → 0`) each new message

---

## Inputs & Textarea

```css
background: #2A2A35;
border: 1px solid #1F1F2E;
border-radius: 0.75rem;
color: #FAF8F5;
transition: border-color 200ms ease;

&:focus {
  border-color: #C9A84C;
  outline: none;
  box-shadow: 0 0 0 3px rgba(201, 168, 76, 0.15);
}
```

---

## Status Indicator

"System Operational" pattern from the original prompt — adapted for our health check:
```jsx
<div className="flex items-center gap-2">
  <span className="relative flex h-2 w-2">
    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
  </span>
  <span className="font-mono text-xs text-green-400">SYSTEM OPERATIONAL</span>
</div>
```

---

## What We're NOT Using from the original prompt

- Hero section (full-bleed image, GSAP stagger headline)
- Features section (Shuffler, Typewriter, Scheduler cards)
- Philosophy / Manifesto section
- Protocol sticky-stack scroll section
- Pricing / Membership section
- Footer
- Agent question flow
- Unsplash image integration
