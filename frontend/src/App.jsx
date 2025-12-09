import { useEffect, useState } from 'react'
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Dashboard from './components/Dashboard'
import DatasetDetail from './components/DatasetDetail'
import DatasetWizard from './components/DatasetWizard'
import GraphWizard from './components/GraphWizard'
import Login from './components/Login'

function AppRoutes() {
  const [user, setUser] = useState(null)
  const [menuOpen, setMenuOpen] = useState(true)
  const navigate = useNavigate()

  const fetchSession = () => {
    fetch('/api/session').then(res => res.json()).then(data => {
      setUser(data.username)
      if (data.username) {
        navigate('/')
      }
    })
  }

  useEffect(() => {
    fetchSession()
  }, [])

  if (!user) {
    return <Login onLoggedIn={fetchSession} />
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar open={menuOpen} setOpen={setMenuOpen} user={{ username: user }} />
      <div className="flex-1 p-4 transition-all">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets/new" element={<DatasetWizard />} />
          <Route path="/datasets/:name" element={<DatasetDetail />} />
          <Route path="/graphs/new" element={<GraphWizard />} />
        </Routes>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}
