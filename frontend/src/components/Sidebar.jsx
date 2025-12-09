import { Link } from 'react-router-dom'
import { HamburgerMenuIcon, AvatarIcon } from '@radix-ui/react-icons'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'

export default function Sidebar({ open, setOpen, user }) {
  const email = user?.email || `${user?.username || ''}@example.com`
  const logout = () => {
    fetch('/api/logout', { method: 'POST' }).then(() => window.location.reload())
  }
  return (
    <div className={`flex flex-col justify-between bg-white border-r h-screen transition-all duration-300 ${open ? 'w-60' : 'w-16'}`}>
      <div>
        <div className="flex items-center justify-between p-4">
          <Link to="/" className="font-bold text-lg">{open ? 'Datacreek' : 'DC'}</Link>
          <button onClick={() => setOpen(!open)} aria-label="toggle menu">
            <HamburgerMenuIcon />
          </button>
        </div>
        <nav className="px-4 space-y-2 text-sm">
          <Link to="/datasets/new" className="block py-1">Nouveau dataset</Link>
          <Link to="/graphs/new" className="block py-1">Nouveau graph</Link>
          <Link to="/marketplace" className="block py-1">Marketplace</Link>
          <div className="h-px bg-gray-200 my-2" />
          <Link to="/" className="block py-1">Mes datasets</Link>
          <Link to="/documents" className="block py-1">Mes documents</Link>
        </nav>
      </div>
      <div className="p-4">
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <button className="flex items-center gap-2 w-full" aria-label="Account">
              <AvatarIcon className="w-6 h-6" />
              {open && <span className="font-medium">{user?.username}</span>}
            </button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content side="top" align="start" className="bg-white rounded shadow p-2">
            <div className="px-2 py-1">
              <div className="font-medium">{user?.username}</div>
              <div className="text-xs text-gray-500">{email}</div>
            </div>
            <div className="h-px bg-gray-200 my-1" />
            <DropdownMenu.Item className="px-2 py-1 cursor-pointer">Setting</DropdownMenu.Item>
            <DropdownMenu.Item className="px-2 py-1 cursor-pointer" onSelect={logout}>Deconnexion</DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
    </div>
  )
}
