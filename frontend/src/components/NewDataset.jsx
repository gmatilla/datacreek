import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, CardHeader, CardContent, Button } from './ui'

export default function NewDataset() {
  const [name, setName] = useState('')
  const [type, setType] = useState('qa')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function submit(e) {
    e.preventDefault()
    setError('')
    const res = await fetch('/api/datasets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, dataset_type: type })
    })
    if (res.ok) {
      navigate(`/datasets/${name}`)
    } else {
      const data = await res.json().catch(() => ({}))
      setError(data.error || 'Error creating dataset')
    }
  }

  return (
    <div className="max-w-sm mx-auto">
      <Card>
        <CardHeader>New Dataset</CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-4">
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
            {error && <p className="text-red-600 text-sm">{error}</p>}
            <Button type="submit" className="w-full">Create</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
