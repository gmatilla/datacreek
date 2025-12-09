import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRightIcon } from '@radix-ui/react-icons'
import { Card, CardHeader, CardContent, Progress, Button } from './ui'

export default function Dashboard() {
  const [datasets, setDatasets] = useState([])
  const [tasks, setTasks] = useState({})

  useEffect(() => {
    fetch('/api/datasets')
      .then(r => r.json())
      .then(names => Promise.all(names.map(n => fetch(`/api/datasets/${n}`).then(r => r.json()))))
      .then(setDatasets)
    const intv = setInterval(() => {
      fetch('/api/tasks').then(r => r.json()).then(setTasks)
    }, 2000)
    return () => clearInterval(intv)
  }, [])

  const activities = [
    ...Object.entries(tasks)
      .filter(([, t]) => t.status === 'running')
      .map(([id, t]) => ({ name: t.label, msg: t.status })),
    ...datasets.flatMap(ds =>
      ds.history.slice(-3).map(msg => ({ name: ds.name, msg }))
    ),
  ].slice(-5).reverse()

  return (
    <div className="grid md:grid-cols-2 gap-4">
      <Card>
        <CardHeader className="flex items-center justify-between">
          <span>Datasets</span>
          <Button asChild className="h-8 px-3 py-1 text-sm">
            <Link to="/datasets/new">New dataset</Link>
          </Button>
        </CardHeader>
        <CardContent>
          {datasets.length ? (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              {datasets.map(ds => (
                <Link
                  key={ds.name}
                  to={`/datasets/${ds.name}`}
                  className="group border rounded-md p-3 flex flex-col gap-2 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-medium">{ds.name}</div>
                      <div className="text-xs text-gray-500 capitalize">{ds.type}</div>
                    </div>
                    <ArrowRightIcon className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                  <Progress value={ds.quality} />
                  <div className="text-xs text-gray-500 mt-auto">
                    {new Date(ds.created_at).toLocaleString()}
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted">No datasets</p>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>Activity</CardHeader>
        <CardContent>
          {activities.length ? (
            <ul className="space-y-1 text-sm">
              {activities.map((a, i) => (
                <li key={i} className="border-b last:border-none pb-1">
                  <span className="font-medium">{a.name}</span>: {a.msg}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted">No recent activity</p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
