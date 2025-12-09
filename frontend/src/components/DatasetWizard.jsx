import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardHeader, CardContent, Button } from './ui'

function StepIndicator({ step }) {
  const steps = ['Dataset', 'Knowledge Graph', 'Generation']
  return (
    <ol className="flex justify-center gap-4 mb-4 text-sm">
      {steps.map((label, i) => (
        <li key={label} className="flex items-center gap-1">
          <span
            className={`w-5 h-5 rounded-full text-white text-xs flex items-center justify-center ${
              step === i ? 'bg-indigo-600' : 'bg-gray-300'
            }`}
          >
            {i + 1}
          </span>
          <span className={step === i ? 'font-medium' : 'text-gray-500'}>{label}</span>
        </li>
      ))}
    </ol>
  )
}

export default function DatasetWizard() {
  const [step, setStep] = useState(0)
  const [name, setName] = useState('')
  const [type, setType] = useState('qa')
  const [docs, setDocs] = useState([''])
  const [highRes, setHighRes] = useState(false)
  const [ocr, setOcr] = useState(false)
  const [extractEntities, setExtractEntities] = useState(false)
  const [extractFacts, setExtractFacts] = useState(false)
  const [useUnstructured, setUseUnstructured] = useState(true)
  const [graphs, setGraphs] = useState([])
  const [graphName, setGraphName] = useState('')
  const [params, setParams] = useState('')
  const navigate = useNavigate()

  async function submit() {
    const res = await fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, dataset_type: type, graph: graphName || null })
    })
    if (!res.ok) return
    if (!graphName) {
      for (const path of docs.filter(Boolean)) {
        const resp = await fetch(`/api/datasets/${name}/ingest`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              path,
              high_res: highRes,
              ocr,
              use_unstructured: useUnstructured,
              extract_entities: extractEntities,
              extract_facts: extractFacts
            })
        })
        if (!resp.ok) {
          // eslint-disable-next-line no-alert
          alert(`Failed to ingest ${path}`)
        }
      }
    }
    let genParams = {}
    try {
      genParams = params ? JSON.parse(params) : {}
    } catch {
      // ignore parse errors
    }
    const generateRes = await fetch(`/api/datasets/${name}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: genParams })
    })
    if (!generateRes.ok) return
    const { task_id } = await generateRes.json()
    if (task_id) {
      const poll = setInterval(async () => {
        const r = await fetch(`/api/tasks/${task_id}`)
        const data = await r.json()
        if (data.status !== 'running') {
          clearInterval(poll)
          navigate(`/datasets/${name}`)
        }
      }, 1000)
    } else {
      navigate(`/datasets/${name}`)
    }
  }

  function addDocField() {
    setDocs(d => [...d, ''])
  }

  function updateDoc(i, value) {
    setDocs(d => d.map((v, idx) => (idx === i ? value : v)))
  }

  useEffect(() => {
    fetch('/api/graphs').then(r => r.json()).then(setGraphs)
  }, [])

  return (
    <div className="max-w-xl mx-auto">
      <Card>
        <CardHeader>New Dataset</CardHeader>
        <CardContent>
          <StepIndicator step={step} />
          {step === 0 && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-1">Name</label>
                <input
                  className="w-full border rounded p-2"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm mb-1">Type</label>
                <select
                  className="w-full border rounded p-2"
                  value={type}
                  onChange={e => setType(e.target.value)}
                >
                  <option value="qa">qa</option>
                  <option value="tool">tool</option>
                  <option value="document">document</option>
                </select>
              </div>
              <div className="flex justify-end gap-2">
                <Button type="button" disabled={!name} onClick={() => setStep(1)}>
                  Next
                </Button>
              </div>
            </div>
          )}
          {step === 1 && (
            <div className="space-y-4">
              {graphs.length > 0 && (
                <div>
                  <label className="block text-sm mb-1">Use existing graph</label>
                  <select
                    className="w-full border rounded p-2"
                    value={graphName}
                    onChange={e => setGraphName(e.target.value)}
                  >
                    <option value="">None</option>
                    {graphs.map(g => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                </div>
              )}
              {!graphName && (
                <div className="space-y-2">
                  {docs.map((d, i) => (
                    <input
                      key={i}
                      className="w-full border rounded p-2"
                      placeholder="Document path or URL"
                      value={d}
                      onChange={e => updateDoc(i, e.target.value)}
                    />
                  ))}
                  <Button type="button" onClick={addDocField}>Add document</Button>
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-1 text-sm">
                      <input
                        type="checkbox"
                        checked={highRes}
                        onChange={e => setHighRes(e.target.checked)}
                      />
                      High-res PDF
                    </label>
                    <label className="flex items-center gap-1 text-sm">
                      <input
                        type="checkbox"
                        checked={ocr}
                        onChange={e => setOcr(e.target.checked)}
                      />
                      OCR
                    </label>
                    <label className="flex items-center gap-1 text-sm">
                      <input
                        type="checkbox"
                        checked={extractEntities}
                        onChange={e => setExtractEntities(e.target.checked)}
                      />
                      Entities
                    </label>
                    <label className="flex items-center gap-1 text-sm">
                      <input
                        type="checkbox"
                        checked={extractFacts}
                        onChange={e => setExtractFacts(e.target.checked)}
                      />
                      Facts
                    </label>
                    <label className="flex items-center gap-1 text-sm">
                      <input
                        type="checkbox"
                        checked={useUnstructured}
                        onChange={e => setUseUnstructured(e.target.checked)}
                      />
                      Unstructured
                    </label>
                  </div>
                </div>
              )}
              <div className="flex justify-between gap-2">
                <Button type="button" onClick={() => setStep(0)}>Back</Button>
                <Button type="button" onClick={() => setStep(2)}>Next</Button>
              </div>
            </div>
          )}
          {step === 2 && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm mb-1">Generation parameters</label>
                <textarea
                  className="w-full border rounded p-2"
                  rows="3"
                  value={params}
                  onChange={e => setParams(e.target.value)}
                />
              </div>
              <div className="flex justify-between gap-2">
                <Button type="button" onClick={() => setStep(1)}>Back</Button>
                <Button type="button" onClick={submit}>Create Dataset</Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
