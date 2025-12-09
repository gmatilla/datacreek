import { useState } from 'react'

export default function Login({ onLoggedIn }) {
  const [mode, setMode] = useState('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')
  const [apiKey, setApiKey] = useState('')

  async function submit(e) {
    e.preventDefault()
    const endpoint = mode === 'login' ? '/api/login' : '/api/register'
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    })
    const data = await res.json()
    if (res.ok) {
      setMessage(data.message)
      if (data.api_key) setApiKey(data.api_key)
      onLoggedIn()
    } else {
      setMessage(data.error)
    }
  }

  return (
    <div className="max-w-sm mx-auto mt-10">
      <h1 className="text-2xl font-bold mb-4 text-center">Datacreek</h1>
      <form onSubmit={submit} className="space-y-4 bg-white p-4 rounded shadow">
        <input
          className="w-full border p-2 rounded"
          placeholder="Username"
          value={username}
          onChange={e => setUsername(e.target.value)}
        />
        <input
          type="password"
          className="w-full border p-2 rounded"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />
        <button className="w-full bg-blue-600 text-white py-2 rounded" type="submit">
          {mode === 'login' ? 'Login' : 'Register'}
        </button>
      </form>
      <p className="text-center mt-2">
        <button className="text-blue-600" onClick={() => setMode(mode === 'login' ? 'register' : 'login')}>
          {mode === 'login' ? 'Create account' : 'Have an account? Login'}
        </button>
      </p>
      {apiKey && (
        <p className="mt-4 text-center text-sm">API key: <code>{apiKey}</code></p>
      )}
      {message && <p className="mt-4 text-center">{message}</p>}
    </div>
  )
}
