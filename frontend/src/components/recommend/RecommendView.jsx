import { useState } from 'react'
import RecommendationCard from './RecommendationCard'
import { getRecommendations } from '../../api'

const SAMPLE_IDS = ['1697', '1668', '1064', '1534', '1067']

export default function RecommendView() {
  const [customerId, setCustomerId] = useState('')
  const [topN, setTopN] = useState(3)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    const id = customerId.trim()
    if (!id) return
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const data = await getRecommendations(id, topN)
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-full flex items-start justify-center px-4 overflow-y-auto py-8">
      <div className="w-full max-w-lg flex flex-col gap-6">
        {/* Header */}
        <div>
          <h2 className="text-[#FAF8F5] font-semibold tracking-tight text-lg">Recommendations</h2>
          <p className="text-xs text-[#6B6B7B] mt-0.5">Content-based personalisation engine</p>
        </div>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">

          {/* Customer ID */}
          <div className="flex flex-col gap-2">
            <label className="text-xs text-[#6B6B7B] tracking-wide uppercase">Customer ID</label>
            <input
              type="text"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              placeholder="Enter customer ID..."
              className="bg-[#2A2A35] text-[#FAF8F5] placeholder-[#6B6B7B] text-sm
                px-4 py-3 rounded-xl border border-[#1F1F2E]
                focus:outline-none focus:border-[#C9A84C] transition-all duration-200"
              style={{ boxShadow: customerId ? '0 0 0 3px rgba(201,168,76,0.1)' : 'none' }}
            />
            {/* Sample IDs */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-[#6B6B7B]">Try:</span>
              {SAMPLE_IDS.map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setCustomerId(id)}
                  className="lift text-xs text-[#C9A84C] hover:text-[#FAF8F5]
                    bg-[#C9A84C]/10 hover:bg-[#C9A84C]/20
                    border border-[#C9A84C]/20 px-2.5 py-1 rounded-full transition-all duration-200"
                  style={{ fontFamily: "'JetBrains Mono', monospace" }}
                >
                  {id}
                </button>
              ))}
            </div>
          </div>

          {/* Top N */}
          <div className="flex flex-col gap-2">
            <label className="text-xs text-[#6B6B7B] tracking-wide uppercase">Results</label>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => setTopN(n)}
                  className={`btn-magnetic w-10 h-10 rounded-xl text-sm font-medium transition-all duration-200
                    ${topN === n
                      ? 'bg-[#C9A84C] text-[#0D0D12]'
                      : 'bg-[#2A2A35] text-[#6B6B7B] border border-[#1F1F2E] hover:text-[#FAF8F5]'
                    }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading || !customerId.trim()}
            className="btn-magnetic relative overflow-hidden group
              bg-[#C9A84C] disabled:bg-[#2A2A35] disabled:text-[#6B6B7B]
              text-[#0D0D12] font-medium px-5 py-3 rounded-xl text-sm
              disabled:cursor-not-allowed transition-colors duration-200"
          >
            <span className="absolute inset-0 bg-[#FAF8F5] translate-y-full group-hover:translate-y-0
              transition-transform duration-300 ease-out" />
            <span className="relative z-10">
              {loading ? 'Analysing...' : 'Get Recommendations'}
            </span>
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm px-4 py-3 rounded-2xl">
            {error}
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="flex flex-col gap-3">
            <p className="text-xs text-[#6B6B7B]">
              {results.recommendations.length} recommendations for{' '}
              <span style={{ fontFamily: "'JetBrains Mono', monospace" }} className="text-[#C9A84C]">
                {results.customer_id}
              </span>
            </p>
            {results.details.map((rec, i) => (
              <RecommendationCard key={rec.service_id} rec={rec} rank={i + 1} delay={i * 0.15} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
