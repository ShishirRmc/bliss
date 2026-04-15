import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

const SUGGESTIONS = [
  'How much does a deep tissue massage cost?',
  'Can I reschedule my booking?',
  'What is the cancellation policy?',
  'I want to cancel my appointment',
]

export default function MessageList({ messages, loading, onSuggest }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-4">
      {messages.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center gap-6 py-16">
          <p style={{ fontFamily: "'Playfair Display', serif", fontStyle: 'italic' }}
            className="text-3xl text-[#FAF8F5]/30 text-center">
            How can I help you today?
          </p>
          <div className="flex flex-col gap-2 w-full max-w-md">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => onSuggest(s)}
                className="lift text-left text-sm text-[#6B6B7B] hover:text-[#FAF8F5]
                  bg-[#2A2A35] hover:bg-[#2A2A35]/80 border border-[#1F1F2E]
                  hover:border-[#C9A84C]/30 px-4 py-3 rounded-2xl transition-all duration-200"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {loading && (
        <div className="flex items-start">
          <div className="bg-[#2A2A35] border border-[#1F1F2E] px-4 py-3 rounded-2xl rounded-bl-sm flex gap-1.5 items-center">
            {[0, 150, 300].map((delay) => (
              <span
                key={delay}
                className="w-1.5 h-1.5 bg-[#C9A84C] rounded-full animate-bounce"
                style={{ animationDelay: `${delay}ms` }}
              />
            ))}
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
