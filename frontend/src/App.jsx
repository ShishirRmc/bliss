import { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import ChatView from './components/chat/ChatView'
import RecommendView from './components/recommend/RecommendView'
import { healthCheck } from './api'

export default function App() {
  const [view, setView] = useState('chat')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
  }, [])

  return (
    <div className="h-screen flex flex-col bg-[#0D0D12] text-[#FAF8F5] overflow-hidden">
      <Navbar health={health} activeView={view} onChangeView={setView} />
      {/* offset for fixed navbar */}
      <main className="flex-1 overflow-hidden mt-14">
        {view === 'chat' ? <ChatView /> : <RecommendView />}
      </main>
    </div>
  )
}
