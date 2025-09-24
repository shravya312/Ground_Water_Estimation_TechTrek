import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import { auth } from './firebase'
import Landing from './pages/Landing'
import Groundwater from './pages/Groundwater'
import Resources from './pages/Resources'
import Chat1 from './pages/Chat1'
import AtHome from './pages/AtHome'
import InYourCommunity from './pages/InYourCommunity'
import GroundwaterDemo from './components/GroundwaterDemo'

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
        <Route path="/chat" element={user ? <Chat1 /> : <Navigate to="/" replace />} />
        <Route path="/demo" element={<GroundwaterDemo />} />
        <Route path="/groundwater" element={<Groundwater />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/at-home" element={<AtHome />} />
        <Route path="/in-your-community" element={<InYourCommunity />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App


