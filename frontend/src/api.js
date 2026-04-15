const BASE_URL = 'http://localhost:8000'

export async function healthCheck() {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error('API unreachable')
  return res.json()
}

export async function sendMessage(message, sessionId) {
  const res = await fetch(`${BASE_URL}/chatbot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!res.ok) throw new Error('Chatbot request failed')
  return res.json()
}

export async function getRecommendations(customerId, topN = 3) {
  const res = await fetch(`${BASE_URL}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: customerId, top_n: topN }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Recommendation request failed')
  }
  return res.json()
}
