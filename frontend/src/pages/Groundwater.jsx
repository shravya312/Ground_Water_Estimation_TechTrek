import { Link } from 'react-router-dom'

function Groundwater() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="m-4 p-3 rounded-xl bg-gradient-card backdrop-blur-sm border border-dark-border shadow-glass animate-float-up">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-secondary-300/35 to-accent-300/35 flex items-center justify-center">
              💧
            </div>
            <span className="font-bold text-dark-text-primary">Ground Water Companion</span>
          </div>
          <nav className="flex gap-2">
            <Link to="/" className="px-3 py-1.5 rounded-lg hover:bg-secondary-300/18 transition-colors">Home</Link>
            <a href="#basics" className="px-3 py-1.5 rounded-lg hover:bg-secondary-300/18 transition-colors">Basics</a>
          </nav>
        </div>
      </header>

      <section className="bg-gradient-to-br from-secondary-300/85 to-secondary-400/75 py-12 px-4" id="basics">
        <div className="w-full max-w-7xl mx-auto text-white">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight">Groundwater Basics</h1>
        </div>
      </section>

      <main className="flex-1 px-4 pb-6">
        <section className="w-full max-w-7xl mx-auto my-6">
          <div className="bg-gradient-to-b from-secondary-300/22 to-accent-300/22 border border-secondary-300/35 rounded-2xl p-6">
            <div className="space-y-4 text-dark-text-primary">
              <p className="text-lg">Over 70% of the Earth's surface is covered in water.</p>
              <p className="text-lg">But of that water, just 1% is readily available for human use, and of that 1%, 99% of it is stored beneath our feet as groundwater.</p>
              <p className="text-lg">We all rely on groundwater in some way, so it's important that we understand this vital resource.</p>
            </div>
          </div>
        </section>

        <section className="w-full max-w-7xl mx-auto my-6 grid lg:grid-cols-[1.5fr_0.5fr] gap-6 items-center">
          <div className="space-y-4">
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-dark-text-primary">What Is Groundwater?</h2>
            <div className="space-y-4 text-dark-text-primary">
              <p className="text-lg leading-relaxed">
                <strong>Groundwater</strong> is the water found underground in the cracks and spaces in soil, sand and rock. It is stored in and moves slowly through geologic formations of soil, sand and rocks called aquifers.
              </p>
              <p className="text-lg leading-relaxed">
                Groundwater is used for drinking water by millions of people, and is critical for irrigation. Managing groundwater sustainably helps ensure availability during dry periods.
              </p>
            </div>
          </div>
          <div className="flex justify-center">
            <img 
              className="w-full h-auto max-h-80 object-cover rounded-xl border border-secondary-300/25 shadow-xl" 
              alt="Lake and grassland" 
              src="https://images.unsplash.com/photo-1473448912268-2022ce9509d8?q=80&w=1400&auto=format&fit=crop" 
            />
          </div>
        </section>

        <section className="w-full max-w-7xl mx-auto my-6 grid lg:grid-cols-[0.5fr_1.5fr] gap-6 items-center">
          <div className="flex justify-center">
            <img 
              className="w-full h-auto max-h-80 object-cover rounded-xl border border-secondary-300/25 shadow-xl" 
              alt="Hydrologic cycle diagram" 
              src="https://i.pinimg.com/736x/6b/6a/8b/6b6a8b4023ae7cc701eba4721db8dcbb.jpg" 
            />
          </div>
          <div className="space-y-4">
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-dark-text-primary">The Hydrologic Cycle</h2>
            <div className="space-y-4 text-dark-text-primary">
              <p className="text-lg leading-relaxed">
                Water is always on the move. Since the earth was formed, it has been endlessly circulating through the <strong>hydrologic cycle</strong>.
              </p>
              <p className="text-lg leading-relaxed">
                Groundwater is an important part of this continuous cycle as water evaporates, forms clouds, returns to earth as precipitation, and recharges aquifers.
              </p>
              <Link 
                to="/" 
                className="inline-block px-6 py-3 bg-dark-surface text-dark-background border border-secondary-300 rounded-lg font-semibold hover:bg-primary-500 hover:text-white transition-all duration-200"
              >
                Back to home
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="m-4 p-3 rounded-xl bg-gradient-card backdrop-blur-sm border border-dark-border shadow-glass">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-secondary-300/35 to-accent-300/35 flex items-center justify-center">
              💧
            </div>
            <span className="font-semibold text-sm text-dark-text-primary">Ground Water Companion</span>
          </div>
          <div className="flex items-center gap-3">
            <Link to="/" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Home</Link>
            <a href="#basics" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Basics</a>
            <a href="mailto:support@example.com" className="text-sm text-dark-text-primary hover:text-primary-300 transition-colors">Support</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default Groundwater


