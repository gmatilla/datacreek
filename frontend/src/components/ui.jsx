import { Slot } from '@radix-ui/react-slot'

export function Card({ children, className }) {
  return <div className={`bg-white rounded shadow ${className || ''}`}>{children}</div>
}
export function CardHeader({ children, className }) {
  return <div className={`border-b px-4 py-2 font-semibold ${className || ''}`}>{children}</div>
}
export function CardContent({ children, className }) {
  return <div className={`p-4 ${className || ''}`}>{children}</div>
}

export function Progress({ value, className }) {
  return (
    <div className={`w-full bg-gray-200 rounded h-2 ${className || ''}`}>
      <div className="bg-green-500 h-2 rounded" style={{ width: `${value}%` }} />
    </div>
  )
}

export function Button({ className, asChild = false, ...props }) {
  const Comp = asChild ? Slot : 'button'
  return (
    <Comp
      className={`inline-flex items-center justify-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:opacity-50 ${className || ''}`}
      {...props}
    />
  )
}
