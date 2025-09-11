import { Link } from 'react-router-dom'
import { useState } from 'react'
import ProtectGW from './ProtectGW.jpg'

function InYourCommunity() {
  const [openSection, setOpenSection] = useState(null);

  const toggleSection = (section) => {
    setOpenSection(openSection === section ? null : section);
  };

  return (
    <div className="landing-root">
      <header className="landing-header glass fade-in-up">
        <div className="landing-header-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <nav className="landing-nav">
          <Link to="/">Home</Link>
          <Link to="/resources">Resources</Link>
        </nav>
      </header>

      <section className="res-hero hero-content-split" id="in-your-community">
        <div className="res-hero-innerac">
          <h1>Protecting Groundwater in Community</h1>
         
        </div>
      </section>

      <main className="res-content">
        <section className="community-main-content">
          <div className="community-intro-split">
            <div className="community-intro-text">
              <p>Groundwater is a primary source of water for communities all over the world. These communities rely on groundwater as water for drinking and other domestic uses, agriculture, industry, and more.</p>
              <p>A clean, sustainable groundwater supply is essential for communities to grow and thrive, making groundwater protection of utmost importance.</p>
              <p>There are many ways your community can help protect its groundwater supply. Get yourself and your community involved as part of the solution to preserving this vital resource!</p>
            </div>
            <div className="community-intro-image">
              <img src={ProtectGW} alt="Community Groundwater" />
            </div>
          </div>
          <div className="community-accordion-container">
            <div className="accordion-item">
              <button className="accordion-header" onClick={() => toggleSection('awareness')}>
                
                <span>Raising Community Awareness</span>
                <span className={`accordion-icon ${openSection === 'awareness' ? 'open' : ''}`}>+</span>
              </button>
              {openSection === 'awareness' && (
                <div className="accordion-content">
                  <p>Organize poster-making, street plays, or awareness campaigns to educate people about groundwater conservation.</p>
                  <p>Encourage residents and students to pledge to conserve water and avoid over-extraction.</p>
                </div>
              )}
            </div>

            <div className="accordion-item">
              <button className="accordion-header" onClick={() => toggleSection('projects')}>
                <span>Community Service Project Ideas</span>
                <span className={`accordion-icon ${openSection === 'projects' ? 'open' : ''}`}>+</span>
              </button>
              {openSection === 'projects' && (
                <div className="accordion-content">
                  <p>You CAN make a difference in your town by educating people about the water they drink every day! Here are some ideas for projects that kids and adults can do in their hometown. The ideas listed here are just to start you thinking; you are encouraged to develop your own!</p>
                  <h3>Groundwater Education Activities</h3>
                  <ul>
                    <li>Teach the people in your community where their drinking water comes from, why it is vulnerable to contamination, and provide ideas for ways they can protect their water. Find out more about source water assessment and protection.</li>
                    <li>Host a Test Your Well event for your community. This event allows well owners to have their water screened for common contaminants such as nitrates and offers opportunities to raise awareness on pollution prevention.</li>
                    <li>Hold a “Mini-Groundwater Festival” for local residents. Teams could demonstrate groundwater concepts using activities in “Making Discoveries,” The Groundwater Gazette, or by creating their own! Topics could include the water cycle, groundwater basics, local groundwater, and groundwater in other states and countries.</li>
                    <li>Lead a groundwater education campaign. Activities could include: speaking to a local government board or council; creating and distributing educational posters, brochures, and newsletters throughout the community; being interviewed on a local TV station; and writing and recording a public service announcement for a local radio station.</li>
                  </ul>
                  <h3>Water Conservation Activities</h3>
                  <ul>
                    <li>Design and install a rain garden for your school to capture rainwater and reduce stormwater pollution. Ask your local nursery for design help and the donation of plants.</li>
                    <li>Check your home/school/business for leaky faucets, showerheads, and toilets. Calculate the amount of wasted water and have the leaks repaired.</li>
                    <li>Research the availability and cost of water conservation equipment for the home/school/business. Replace appliances with environmentally friendly models when appropriate.</li>
                  </ul>
                  <h3>Pollution Prevention Awareness Activities</h3>
                  <ul>
                    <li>Prevent groundwater pollution from occurring in the first place by teaching people how to keep water safe for all of us!</li>
                    <li>Stencil messages on storm drains to help prevent chemicals and oil from being dumped down them. These chemicals pollute both surface and groundwater and are dangerous to aquatic plants and animals.</li>
                    <li>Design and distribute posters or fliers in the community which educate residents about the hazards of abandoned wells and how they should be properly filled.</li>
                    <li>Share conservation messages that encourage the wise use of groundwater supplies. Examples could include performing a play or writing a comic book about water conservation.</li>
                  </ul>
                  <h3>Community Water History Awareness Activities</h3>
                  <ul>
                    <li>Investigate and share the history of your community’s successes and challenges in finding adequate and safe water supplies.</li>
                    <li>Check local newspaper archives (at the newspaper’s office or at the library) to find stories from the past about your community’s water supply, including natural disasters which may have affected the supply.</li>
                    <li>Interview senior residents about memories of your community finding safe water supplies through the years.</li>
                    <li>Make a map showing your community’s water supply source. Display maps at schools, public libraries, and grocery stores.</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="accordion-item">
              <button className="accordion-header" onClick={() => toggleSection('wellhead')}>
                <span>Wellhead Protection</span>
                <span className={`accordion-icon ${openSection === 'wellhead' ? 'open' : ''}`}>+</span>
              </button>
              {openSection === 'wellhead' && (
                <div className="accordion-content">
                  <p>Strategies to prevent contamination of groundwater sources</p>
                  <p>Wellhead protection means protecting the area surrounding public and private drinking water supply wells to ensure a safe, clean, and sustainable drinking water source. Groundwater serves as the primary source of drinking water for many communities, and protecting it is essential for public health and environmental sustainability.</p>
                  <p>With expanding urban development, growing populations, and intensive agricultural practices, the risk of groundwater contamination is increasing. Proactive wellhead protection strategies help prevent pollutants from entering drinking water supplies, reduce treatment costs, and safeguard community health.</p>
                  <p>Each state and community may have a slightly different approach, but the following are the key components of a wellhead protection program:</p>
                  <h3>Key Strategies for Wellhead Protection</h3>
                  <ol>
                    <li><b>Delineating the Wellhead Protection Area (WHPA)</b>
                      <p>A map is created to define the land area surrounding a well that could influence groundwater quality.</p>
                      <p>It considers:</p>
                      <ul>
                        <li>Groundwater flow direction</li>
                        <li>Time-of-travel zones (how long it takes water to reach the well)</li>
                        <li>Aquifer geology</li>
                        <li>Well pumping capacity</li>
                      </ul>
                      <p>This mapping helps identify critical zones where extra precautions are needed.</p>
                    </li>
                    <li><b>Conducting a Potential Contaminant Source Inventory</b>
                      <p>An inventory of potential contamination sources within the WHPA is conducted using:</p>
                      <ul>
                        <li>State regulatory databases</li>
                        <li>Local land-use maps</li>
                        <li>On-the-ground inspections</li>
                      </ul>
                      <p>Common sources of contamination include:</p>
                      <ul>
                        <li>Agricultural runoff (pesticides, fertilizers)</li>
                        <li>Industrial waste discharge</li>
                        <li>Leaking septic tanks</li>
                        <li>Landfills, gas stations, and chemical storage facilities</li>
                      </ul>
                      <p>This helps prioritize high-risk locations for monitoring and management.</p>
                    </li>
                    <li><b>Contaminant Source Management</b>
                      <p>Once risks are identified, communities take steps to prevent pollutants from entering groundwater by:</p>
                      <ul>
                        <li>Enforcing zoning regulations and land-use restrictions</li>
                        <li>Purchasing sensitive land for protection</li>
                        <li>Encouraging best management practices in agriculture and industry</li>
                        <li>Establishing conservation easements</li>
                        <li>Promoting voluntary compliance by businesses and residents</li>
                      </ul>
                      <p>These measures limit potential contamination before it reaches drinking water sources.</p>
                    </li>
                  </ol>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>

      <footer className="landing-footer glass">
        <div className="footer-left">
          <div className="logo-circle">💧</div>
          <span className="brand">Ground Water Companion</span>
        </div>
        <div className="footer-right">
          <Link to="/">Home</Link>
          <Link to="/resources">Resources</Link>
          <a href="mailto:support@example.com">Support</a>
        </div>
      </footer>
    </div>
  );
}

export default InYourCommunity
