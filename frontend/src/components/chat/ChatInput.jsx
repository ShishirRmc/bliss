import { useState } from 'react'

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e)
  }

  const canSend = !disabled && value.trim()

  return (
    <form
      onSubmit={handleSubmit}
      className="px-4 py-4 border-t border-[#1F1F2E] flex gap-3 items-end"
    >
      <textarea
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask about pricing, policies, or manage your bookings..."
        disabled={disabled}
        className="flex-1 bg-[#2A2A35] text-[#FAF8F5] placeholder-[#6B6B7B] text-sm
          px-4 py-3 rounded-xl border border-[#1F1F2E] resize-none
          focus:outline-none focus:border-[#C9A84C]
          disabled:opacity-40 transition-all duration-200"
        style={{ boxShadow: value ? '0 0 0 3px rgba(201,168,76,0.1)' : 'none' }}
      />
      <button
        type="submit"
        disabled={!canSend}
        className="btn-magnetic relative overflow-hidden group
          bg-[#C9A84C] disabled:bg-[#2A2A35] disabled:text-[#6B6B7B]
          text-[#0D0D12] font-medium px-5 py-3 rounded-xl text-sm
          disabled:cursor-not-allowed transition-colors duration-200"
      >
        <span className="absolute inset-0 bg-[#FAF8F5] translate-y-full group-hover:translate-y-0
          transition-transform duration-300 ease-out" />
        <span className="relative z-10">Send</span>
      </button>
    </form>
  )
}
