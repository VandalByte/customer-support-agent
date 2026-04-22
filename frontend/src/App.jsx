import { useEffect, useMemo, useState } from 'react'
import './App.css'

const fetchJsonWithTimeout = async (url, { timeoutMs = 4000, ...opts } = {}) => {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const resp = await fetch(url, { ...opts, signal: controller.signal })
    const text = await resp.text()
    let data = null
    try {
      data = text ? JSON.parse(text) : null
    } catch {
      data = null
    }
    return { ok: resp.ok, status: resp.status, data, text }
  } finally {
    clearTimeout(timer)
  }
}

function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState('strict')
  const [temperature, setTemperature] = useState(0.2)
  const [maxTokens, setMaxTokens] = useState(150)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [showPrompt, setShowPrompt] = useState(false)
  const [apiBase, setApiBase] = useState('')
  const [health, setHealth] = useState({
    loading: true,
    ok: false,
    model: '',
    policies: 0,
  })
  const [copied, setCopied] = useState(false)

  const presets = useMemo(
    () => ({
      strict: { temperature: 0.2, max_tokens: 150, label: 'Strict (Policy)' },
      balanced: { temperature: 0.5, max_tokens: 180, label: 'Balanced' },
      friendly: { temperature: 0.7, max_tokens: 200, label: 'Friendly' },
    }),
    []
  )

  const examples = useMemo(
    () => [
      'My product arrived late and damaged. Can I get a refund?',
      'I received the wrong size. How do I exchange it?',
      'The courier shows delivered, but I did not receive the package. What should I do?',
    ],
    []
  )

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      try {
        const candidates = ['', 'http://localhost:8000', 'http://127.0.0.1:8000']
        for (const base of candidates) {
          const url = `${base}/api/health`
          try {
            const resp = await fetchJsonWithTimeout(url, { timeoutMs: 4000 })
            if (resp.ok && resp.data?.ok) {
              if (cancelled) return
              setApiBase(base)
              setHealth({
                loading: false,
                ok: true,
                model: String(resp.data?.model || ''),
                policies: Number(resp.data?.policies || 0),
              })
              return
            }
          } catch {
            // Try next candidate
          }
        }
        if (cancelled) return
        setHealth({ loading: false, ok: false, model: '', policies: 0 })
      } catch {
        if (cancelled) return
        setHealth({ loading: false, ok: false, model: '', policies: 0 })
      }
    }
    run()
    return () => {
      cancelled = true
    }
  }, [])

  const applyPreset = (nextMode) => {
    const preset = presets[nextMode]
    if (!preset) return
    setMode(nextMode)
    setTemperature(preset.temperature)
    setMaxTokens(preset.max_tokens)
  }

  const resetToPreset = () => applyPreset(mode)

  const clearAll = () => {
    setQuery('')
    setError('')
    setResult(null)
  }

  const copyResponse = async () => {
    const text = result?.response || ''
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 900)
    } catch {
      setError('Copy failed (browser permission).')
    }
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)

    const trimmed = query.trim()
    if (!trimmed) {
      setError('Please enter a customer complaint.')
      return
    }

    setLoading(true)
    try {
      const resp = await fetch(`${apiBase}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: trimmed,
          mode,
          temperature: Number(temperature),
          max_tokens: Number(maxTokens),
        }),
      })

      if (!resp.ok) {
        const text = await resp.text()
        throw new Error(text || `Request failed (${resp.status})`)
      }

      const data = await resp.json()
      setResult(data)
    } catch (err) {
      setError(err?.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="titleRow">
          <div>
            <h1>AI-Assisted Support Reply</h1>
            {/* <p className="subtitle">BM25 policy retrieval + GPT-4o mini drafting</p> */}
          </div>
          <div className="status">
            {health.loading ? (
              <span className="pill muted">Connecting...</span>
            ) : health.ok ? (
              <span className="pill ok">
                Connected - {health.model || 'model'} - {health.policies} policies
              </span>
            ) : (
              <span className="pill warn">Backend not reachable</span>
            )}
          </div>
        </div>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>Customer Complaint</h2>
          <form onSubmit={onSubmit} className="form">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='Example: "My product arrived late and damaged. Can I get a refund?"'
              rows={7}
            />

            <div className="chips">
              <div className="chipsLabel">Examples</div>
              <div className="chipsRow">
                {examples.map((ex) => (
                  <button
                    key={ex}
                    type="button"
                    className="chip"
                    onClick={() => setQuery(ex)}
                    disabled={loading}
                    title="Use this example"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            <div className="row">
              <label>
                Mode
                <select
                  value={mode}
                  onChange={(e) => applyPreset(e.target.value)}
                >
                  {Object.entries(presets).map(([key, p]) => (
                    <option key={key} value={key}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Temperature
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                />
              </label>

              <label>
                Max tokens
                <input
                  type="number"
                  min="50"
                  max="500"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                />
              </label>
            </div>

            <div className="row actions">
              <div className="actionsLeft">
                <button type="submit" disabled={loading || !health.ok}>
                  {loading ? 'Generating…' : 'Generate Response'}
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={clearAll}
                  disabled={loading}
                >
                  Clear
                </button>
              </div>
              <div className="actionsRight">
                <button
                  type="button"
                  className="ghost"
                  onClick={resetToPreset}
                  disabled={loading}
                >
                  Reset to preset
                </button>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showPrompt}
                    onChange={(e) => setShowPrompt(e.target.checked)}
                  />
                  Show prompt
                </label>
              </div>
            </div>

            {error ? <div className="error">{error}</div> : null}
          </form>
        </section>

        <section className="panel">
          <div className="panelHeader">
            <h2>AI Response</h2>
            <div className="panelTools">
              <button
                type="button"
                className="ghost"
                onClick={copyResponse}
                disabled={!result?.response}
              >
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
          </div>
          {!result ? (
            <div className="empty">
              Submit a complaint to see the draft response and retrieved policies.
            </div>
          ) : (
            <>
              {result.fallback ? (
                <div className="badge warn">Fallback (no strong policy match)</div>
              ) : (
                <div className="badge ok">Policy-grounded</div>
              )}
              <div className="response">{result.response}</div>

              <div className="meta">
                <span>Mode: {result.mode}</span>
                <span>Temperature: {result.temperature}</span>
                <span>Max tokens: {result.max_tokens}</span>
              </div>

              {showPrompt && result.prompt_used ? (
                <details className="details" open={false}>
                  <summary>Prompt used</summary>
                  <pre>{result.prompt_used}</pre>
                </details>
              ) : null}

              <details className="details" open>
                <summary>Retrieved policy documents</summary>
                {result.retrieved?.length ? (
                  <div className="docs">
                    {result.retrieved.map((d, idx) => (
                      <details key={`${d.title}-${idx}`} className="doc" open={idx === 0}>
                        <summary className="docTitle">
                          <span className="docTitleText">{d.title}</span>
                          <span className="score">score: {d.score.toFixed(3)}</span>
                        </summary>
                        <div className="docBody">{d.content}</div>
                      </details>
                    ))}
                  </div>
                ) : (
                  <div className="empty">No documents retrieved.</div>
                )}
              </details>
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>Tip: increase temperature for a more natural, friendly tone.</span>
      </footer>
    </div>
  )
}

export default App
