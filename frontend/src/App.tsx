import { useEffect, useMemo, useState } from 'react'
import './App.css'

type Health = {
  status: string
  model_loaded: boolean
  model_path: string
}

type Metadata = {
  model_name: string
  model_path: string
  feature_names: string[]
  feature_count: number
  label_map: Record<string, number>
  preprocessing: string
  notes?: string
}

type PredictResponse = {
  predicted_label: number
  predicted_label_name: string
  probability?: number | null
  model_name: string
  feature_count: number
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [meta, setMeta] = useState<Metadata | null>(null)
  const [features, setFeatures] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sample, setSample] = useState<Record<string, number> | null>(null)
  const [inputMode, setInputMode] = useState<'form' | 'json'>('form')
  const [jsonInput, setJsonInput] = useState<string>('')

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (!res.ok) throw new Error(`health ${res.status}`)
      const data = await res.json()
      setHealth(data)
    } catch (e: any) {
      setError(e.message)
    }
  }

  const fetchMeta = async () => {
    try {
      const res = await fetch(`${API_BASE}/metadata`)
      if (!res.ok) throw new Error(`metadata ${res.status}`)
      const data = await res.json()
      setMeta(data)
      // init form
      const init: Record<string, string> = {}
      for (const f of data.feature_names) init[f] = ''
      setFeatures(init)
    } catch (e: any) {
      setError(e.message)
    }
  }

  const loadExample = async () => {
    try {
      const res = await fetch('/example_request.json')
      if (!res.ok) throw new Error('example not found')
      const data = await res.json()
      setSample(data.features)
      const next: Record<string, string> = {}
      for (const [k,v] of Object.entries(data.features)) next[k] = String(v)
      setFeatures(next)
      setError(null)
    } catch (e: any) {
      setError(e.message)
    }
  }

  useEffect(() => {
    fetchHealth()
    fetchMeta()
  }, [])

  const canPredict = useMemo(() => {
    if (!meta) return false
    if (loading) return false
    for (const f of meta.feature_names) {
      if (features[f] === undefined || features[f] === '') return false
    }
    return true
  }, [meta, features, loading])

  const onPredict = async () => {
    if (!meta) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      let payload: Record<string, number> = {}
      if (inputMode === 'json') {
        if (!jsonInput.trim()) throw new Error('Provide JSON input')
        let parsed: any
        try {
          parsed = JSON.parse(jsonInput)
        } catch (e: any) {
          throw new Error('Invalid JSON')
        }
        const feats = parsed.features ? parsed.features : parsed
        if (typeof feats !== 'object' || feats === null) throw new Error('Expected features object')
        // populate form for visibility
        const next: Record<string, string> = { ...features }
        for (const [k, v] of Object.entries(feats)) {
          next[k] = String(v as any)
        }
        setFeatures(next)
        payload = {}
        for (const f of meta.feature_names) {
          if (!(f in feats)) throw new Error(`Missing feature in JSON: ${f}`)
          const num = Number((feats as any)[f])
          if (Number.isNaN(num)) throw new Error(`Invalid number for ${f}`)
          payload[f] = num
        }
      } else {
        payload = {}
        for (const f of meta.feature_names) {
          const v = features[f]
          if (v === undefined || v === '') throw new Error(`Missing value for ${f}`)
          const num = Number(v)
          if (Number.isNaN(num)) throw new Error(`Invalid number for ${f}`)
          payload[f] = num
        }
      }
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: payload }),
      })
      if (!res.ok) {
        let msg = `predict ${res.status}`
        try {
          const j = await res.json()
          msg = j.detail ? JSON.stringify(j.detail) : JSON.stringify(j)
        } catch {}
        throw new Error(msg)
      }
      const data = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const onReset = () => {
    if (!meta) return
    const init: Record<string, string> = {}
    for (const f of meta.feature_names) init[f] = ''
    setFeatures(init)
    setResult(null)
    setError(null)
  }

  const statusOk = health?.model_loaded

  return (
    <div className="app">
      <header className="header">
        <h1>Comparative Evaluation of Hyperparameter Optimization and AutoML Frameworks for Network Intrusion Detection</h1>
        <p>Manual prediction dashboard for CIC-IDS2017 binary classification (BENIGN vs ATTACK)</p>
        <div className="status">
          <span className={`dot ${statusOk ? 'ok' : health ? 'err' : ''}`}></span>
          <span>Backend: {health ? (statusOk ? 'connected (model loaded)' : 'connected (model not loaded)') : 'checking...'}</span>
          {health?.model_path && <span> | {health.model_path}</span>}
        </div>
      </header>

      <section className="section">
        <h2>Actions</h2>
        <div className="actions">
          <button className="btn" onClick={loadExample}>Load Sample (demonstration)</button>
          <button className="btn" onClick={onReset}>Reset Form</button>
          <button className="btn primary" disabled={loading} onClick={onPredict}>
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
        {sample && <p style={{color:'#cbd5e1'}}>Loaded demonstration sample values from repository example.</p>}
      </section>

      <section className="section">
        <h2>Input Mode</h2>
        <div className="actions">
          <button className={`btn ${inputMode === 'form' ? 'primary' : ''}`} onClick={() => setInputMode('form')}>Form</button>
          <button className={`btn ${inputMode === 'json' ? 'primary' : ''}`} onClick={() => setInputMode('json')}>JSON</button>
        </div>
        {inputMode === 'json' && (
          <div style={{display:'flex', flexDirection:'column', gap:'8px'}}>
            <label style={{color:'#cbd5e1'}}>Paste either {'{"features": {...}}'} or just the features object</label>
            <textarea
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              rows={10}
              style={{background:'#0f172a', color:'#e2e8f0', border:'1px solid #334155', borderRadius:'8px', padding:'10px', fontFamily:'monospace'}}
              placeholder='{"features": {"Destination Port": 80, ...}}'
            />
          </div>
        )}
      </section>

      {meta && (
        <section className="section">
          <h2>Expected classes</h2>
          <p>BENIGN (0), ATTACK (1) — based on API metadata</p>
          <p>Features: {meta.feature_count} (exact schema from backend)</p>
        </section>
      )}

      {meta && (
        <section className="section">
          <h2>Network Flow Features</h2>
          <div className="features-grid">
            {meta.feature_names.map((f) => (
              <div className="field" key={f}>
                <label>{f} <span style={{color:'#f59e0b'}}>*</span></label>
                <input
                  type="number"
                  step="any"
                  value={features[f] || ''}
                  onChange={(e) => setFeatures({ ...features, [f]: e.target.value })}
                  placeholder="numeric value"
                />              </div>
            ))}
          </div>
        </section>
      )}

      {(result || error) && (
        <section className="section">
          <h2>Result</h2>
          {error && <p className="error">Error: {error}</p>}
          {result && (
            <div className="result">
              <div className="success">Prediction: <strong>{result.predicted_label_name}</strong> (label {result.predicted_label})</div>
              {result.probability !== null && result.probability !== undefined && (
                <div>Attack probability: {result.probability}</div>
              )}
              <div>Model: {result.model_name} | Features: {result.feature_count}</div>
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export default App

