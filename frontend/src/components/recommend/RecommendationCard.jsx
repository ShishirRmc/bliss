import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'

const CATEGORY_STYLES = {
  body:  { color: '#7DD3FC', bg: 'rgba(125,211,252,0.08)', border: 'rgba(125,211,252,0.2)' },
  face:  { color: '#F9A8D4', bg: 'rgba(249,168,212,0.08)', border: 'rgba(249,168,212,0.2)' },
  combo: { color: '#C9A84C', bg: 'rgba(201,168,76,0.08)',  border: 'rgba(201,168,76,0.2)'  },
  nails: { color: '#86EFAC', bg: 'rgba(134,239,172,0.08)', border: 'rgba(134,239,172,0.2)' },
}

export default function RecommendationCard({ rec, rank, delay = 0 }) {
  const cardRef = useRef(null)
  const barRef = useRef(null)
  const style = CATEGORY_STYLES[rec.category] || CATEGORY_STYLES.body
  const confidencePct = Math.round(rec.confidence * 100)

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Card entrance
      gsap.from(cardRef.current, {
        y: 24, opacity: 0, duration: 0.5, ease: 'power3.out', delay,
      })
      // Bar fill
      gsap.from(barRef.current, {
        scaleX: 0, duration: 0.8, ease: 'power3.out', delay: delay + 0.2,
        transformOrigin: 'left center',
      })
    }, cardRef)
    return () => ctx.revert()
  }, [])

  return (
    <div
      ref={cardRef}
      className="bg-[#2A2A35] border border-[#1F1F2E] rounded-3xl p-5 flex flex-col gap-4
        shadow-[0_8px_32px_rgba(0,0,0,0.4)] hover:-translate-y-0.5
        hover:shadow-[0_12px_40px_rgba(0,0,0,0.5)] transition-all duration-300 cursor-default"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span
            style={{ fontFamily: "'JetBrains Mono', monospace" }}
            className="text-xs text-[#6B6B7B]"
          >
            #{rank}
          </span>
          <span className="text-[#FAF8F5] font-medium tracking-tight">{rec.service}</span>
        </div>
        <span
          className="text-xs px-2.5 py-1 rounded-full font-medium shrink-0"
          style={{ color: style.color, background: style.bg, border: `1px solid ${style.border}` }}
        >
          {rec.category}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="flex items-center gap-3">
        <div className="flex-1 bg-[#0D0D12] rounded-full h-1 overflow-hidden">
          <div
            ref={barRef}
            className="h-1 rounded-full"
            style={{ width: `${confidencePct}%`, background: style.color }}
          />
        </div>
        <span
          style={{ fontFamily: "'JetBrains Mono', monospace", color: style.color }}
          className="text-xs w-9 text-right"
        >
          {confidencePct}%
        </span>
      </div>

      {/* Price */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-[#6B6B7B]">Avg. price</span>
        <span
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
          className="text-sm text-[#FAF8F5]"
        >
          ${rec.avg_price}
        </span>
      </div>

      {rec.note && (
        <span className="text-xs text-[#C9A84C]/60">{rec.note}</span>
      )}
    </div>
  )
}
