import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import ReactMarkdown from 'react-markdown'

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
        {isUser ? message.content : (
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="font-semibold text-[#C9A84C]">{children}</strong>,
              em: ({ children }) => <em className="italic text-[#A0A0B0]">{children}</em>,
              ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 my-2">{children}</ol>,
              ul: ({ children }) => <ul className="list-disc list-inside space-y-1 my-2">{children}</ul>,
              li: ({ children }) => <li className="leading-relaxed">{children}</li>,
              a: ({ href, children }) => (
                <a href={href} className="text-[#C9A84C] underline underline-offset-2 hover:opacity-80" target="_blank" rel="noreferrer">{children}</a>
              ),
              code: ({ children }) => (
                <code style={{ fontFamily: "'JetBrains Mono', monospace" }} className="bg-[#0D0D12] text-[#C9A84C] px-1.5 py-0.5 rounded text-xs">{children}</code>
              ),
              hr: () => <hr className="border-[#1F1F2E] my-2" />,
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
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
