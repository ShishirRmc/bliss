import { useState, useEffect } from 'react'

export default function Navbar({ health, activeView, onChangeView }) {
  const [scrolled, setScrolled] = useState(false)
  const isOnline = health?.status === 'ok'

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 80)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 h-14
        transition-all duration-500 border-b
        ${scrolled
          ? 'bg-[#0D0D12]/90 backdrop-blur-xl border-[#2A2A35]'
          : 'bg-[#0D0D12] border-[#1F1F2E]'
        }`}
    >
      {/* Logo — left */}
      <span className="text-[#FAF8F5] font-semibold tracking-tight text-sm select-none w-32">
        Bliss{' '}
        <span style={{ fontFamily: "'Playfair Display', serif", fontStyle: 'italic' }}
          className="text-[#C9A84C]">
          AI
        </span>
      </span>

      {/* Nav — center */}
      <div className="flex items-center gap-1 bg-[#1A1A28] border border-[#3A3A50] rounded-full px-1.5 py-1.5">
        {[
          { id: 'chat', label: '💬 Chat' },
          { id: 'recommend', label: '✨ Recommend' },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => onChangeView(item.id)}
            className={`px-5 py-1.5 rounded-full text-sm font-medium transition-all duration-200
              ${activeView === item.id
                ? 'bg-[#C9A84C] text-[#0D0D12]'
                : 'text-[#A0A0B8] hover:text-[#FAF8F5]'
              }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* Status — right */}
      <div className="flex items-center gap-2 w-32 justify-end">
        <span className="relative flex h-1.5 w-1.5">
          {isOnline && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
          )}
          <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${isOnline ? 'bg-green-400' : 'bg-red-500'}`} />
        </span>
        <span
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
          className={`text-xs ${isOnline ? 'text-green-400' : 'text-red-400'}`}
        >
          {isOnline ? health.provider.toUpperCase() : 'OFFLINE'}
        </span>
      </div>
    </nav>
  )
}
