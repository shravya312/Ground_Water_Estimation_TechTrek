import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from './firebase'
import Landing from './pages/Landing'
import Groundwater from './pages/Groundwater'
import Resources from './pages/Resources'
import Chat from './pages/Chat'

function App() {
  const [user, setUser] = useState(null)

  useEffect(() => {
    const unsub = onAuthStateChanged(auth, setUser)
    return () => unsub()
  }, [])

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/chat" element={user ? <Chat /> : <Navigate to="/" replace />} />
        <Route path="/groundwater" element={<Groundwater />} />
        <Route path="/resources" element={<Resources />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App


