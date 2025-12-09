import { useEffect, useState } from 'react'
import { NavigationMenu, NavigationMenuList, NavigationMenuItem, NavigationMenuTrigger, NavigationMenuContent } from '@radix-ui/react-navigation-menu'
import { Link } from 'react-router-dom'

export default function Navbar() {
  const [user, setUser] = useState(null)

  useEffect(() => {
    fetch('/api/session').then(r => r.json()).then(d => setUser(d.username))
  }, [])

  const logout = () => {
    fetch('/api/logout', { method: 'POST' }).then(() => window.location.reload())
  }

  return (
    <NavigationMenu className="bg-white shadow px-4 py-2 mb-6">
      <NavigationMenuList className="flex gap-4">
        <NavigationMenuItem>
          <NavigationMenuTrigger className="font-bold">Datacreek</NavigationMenuTrigger>
          <NavigationMenuContent className="p-2 bg-white rounded-md shadow">
            <Link className="block px-2 py-1 hover:bg-gray-100 rounded" to="/">Dashboard</Link>
            <Link className="block px-2 py-1 hover:bg-gray-100 rounded" to="/datasets">Datasets</Link>
          </NavigationMenuContent>
        </NavigationMenuItem>
        {user && (
          <NavigationMenuItem className="ml-auto">
            <NavigationMenuTrigger>{user}</NavigationMenuTrigger>
            <NavigationMenuContent className="p-2 bg-white rounded-md shadow">
              <button className="px-2 py-1 hover:bg-gray-100 rounded w-full text-left" onClick={logout}>Logout</button>
            </NavigationMenuContent>
          </NavigationMenuItem>
        )}
      </NavigationMenuList>
    </NavigationMenu>
  )
}
