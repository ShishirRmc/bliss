import { useState } from 'react'
import MessageList from './MessageList'
import ChatInput from './ChatInput'
import { sendMessage } from '../../api'

const SESSION_ID = `session_${Math.random().toString(36).slice(2)}`

export default function ChatView() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function handleSend(text) {
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setLoading(true)
    try {
      const data = await sendMessage(text, SESSION_ID)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response,
          intent: data.intent,
          sources: data.sources,
        },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '⚠️ Could not reach the API. Is the backend running?' },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex items-center justify-center px-4">
      <div className="w-full max-w-2xl h-[75vh] flex flex-col rounded-3xl border border-[#1F1F2E] bg-[#0D0D12] shadow-[0_24px_80px_rgba(0,0,0,0.6)] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[#1F1F2E] shrink-0 flex items-center justify-between">
          <div>
            <h2 className="text-[#FAF8F5] font-semibold tracking-tight">Chat</h2>
            <p className="text-xs text-[#6B6B7B] mt-0.5">
              RAG-powered assistant · session{' '}
              <span style={{ fontFamily: "'JetBrains Mono', monospace" }} className="text-[#C9A84C]/60">
                {SESSION_ID.slice(-6)}
              </span>
            </p>
          </div>
          <div className="w-2 h-2 rounded-full bg-[#C9A84C]/40" />
        </div>
        <MessageList messages={messages} loading={loading} onSuggest={handleSend} />
        <ChatInput onSend={handleSend} disabled={loading} />
      </div>
    </div>
  )
}
