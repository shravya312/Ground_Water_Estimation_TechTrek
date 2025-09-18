// Google Maps API Loader Utility
let isGoogleMapsLoaded = false;
let loadPromise = null;

export const loadGoogleMaps = () => {
  if (isGoogleMapsLoaded) {
    return Promise.resolve();
  }

  if (loadPromise) {
    return loadPromise;
  }

  loadPromise = new Promise((resolve, reject) => {
    // Check if Google Maps is already loaded
    if (window.google && window.google.maps) {
      isGoogleMapsLoaded = true;
      resolve();
      return;
    }

    // Use hardcoded API key for now
    const apiKey = 'AIzaSyBuSA6XXz7ZmSFonptlXs1ALyNTZfLrf8g';
    
    console.log('Using Google Maps API key:', apiKey.substring(0, 10) + '...');

    // Create script element
    const script = document.createElement('script');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places`;
    script.async = true;
    script.defer = true;
    
    script.onload = () => {
      isGoogleMapsLoaded = true;
      resolve();
    };
    
    script.onerror = () => {
      reject(new Error('Failed to load Google Maps API'));
    };

    // Add script to document
    document.head.appendChild(script);
  });

  return loadPromise;
};

export const isGoogleMapsReady = () => {
  return isGoogleMapsLoaded && window.google && window.google.maps;
};
