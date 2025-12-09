import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardHeader, CardContent, Button } from './ui'

export default function GraphWizard() {
  const [name, setName] = useState('')
  const [docs, setDocs] = useState([''])
  const navigate = useNavigate()

  function addDocField() {
    setDocs(d => [...d, ''])
  }
  function updateDoc(i, value) {
    setDocs(d => d.map((v, idx) => (idx === i ? value : v)))
  }

  async function submit() {
    const res = await fetch('/api/graphs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, documents: docs.filter(Boolean) })
    })
    if (res.ok) {
      navigate('/')
    }
  }

  return (
    <div className="max-w-xl mx-auto">
      <Card>
        <CardHeader>Create Knowledge Graph</CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm mb-1">Name</label>
            <input
              className="w-full border rounded p-2"
              value={name}
              onChange={e => setName(e.target.value)}
              required
            />
          </div>
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
          </div>
          <div className="flex justify-end">
            <Button type="button" onClick={submit} disabled={!name}>Create</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
