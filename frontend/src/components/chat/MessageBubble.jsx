import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const ref = useRef(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from(ref.current, { y: 12, opacity: 0, duration: 0.4, ease: 'power3.out' })
    }, ref)
    return () => ctx.revert()
  }, [])

  return (
    <div ref={ref} className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} gap-1.5`}>
      <div
        className={`max-w-[72%] px-4 py-3 text-sm leading-relaxed
          ${isUser
            ? 'rounded-2xl rounded-br-sm text-[#FAF8F5] border'
            : 'rounded-2xl rounded-bl-sm text-[#FAF8F5] bg-[#2A2A35] border border-[#1F1F2E]'
          }`}
        style={isUser ? {
          background: 'rgba(201,168,76,0.12)',
          borderColor: 'rgba(201,168,76,0.3)',
        } : {}}
      >
        {message.content}
      </div>

      {message.sources?.length > 0 && (
        <div className="flex flex-wrap gap-1 max-w-[72%]">
          {message.sources.map((src, i) => (
            <span
              key={i}
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
              className="text-xs bg-[#0D0D12] text-[#C9A84C] border border-[#C9A84C]/20 px-2 py-0.5 rounded-full"
            >
              {src}
            </span>
          ))}
        </div>
      )}

      {message.intent && !isUser && (
        <span
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
          className="text-xs text-[#6B6B7B]"
        >
          {message.intent.replace(/_/g, ' ')}
        </span>
      )}
    </div>
  )
}
