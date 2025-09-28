# Dropdown System Documentation

## Overview
The dropdown system provides a user-friendly interface for selecting states and districts to analyze groundwater data. The system includes both backend API endpoints and frontend React components.

## Features
- ✅ **37 States** across India
- ✅ **751 Districts** with groundwater data
- ✅ **1,203 Taluks** with detailed groundwater data
- ✅ **Hierarchical Selection** (State → District → Taluk)
- ✅ **Real-time Data Loading**
- ✅ **Ooty Support** (THE NILGIRIS district in Tamil Nadu)
- ✅ **Smart Location Detection**
- ✅ **Responsive Design**
- ✅ **Taluk-level Analysis** (NEW!)

## Backend API Endpoints

### 1. Get All States
```
GET /dropdown/states
```
**Response:**
```json
{
  "success": true,
  "states": ["ANDAMAN AND NICOBAR ISLANDS", "ANDHRA PRADESH", ...],
  "count": 37
}
```

### 2. Get All Districts
```
GET /dropdown/districts
```
**Response:**
```json
{
  "success": true,
  "districts": ["AGAR MALWA", "AGRA", "AHMEDABAD", ...],
  "count": 751
}
```

### 3. Get Districts by State
```
GET /dropdown/districts/{state}
```
**Example:** `/dropdown/districts/TAMILNADU`

**Response:**
```json
{
  "success": true,
  "state": "TAMILNADU",
  "districts": ["ARIYALUR", "CHENGALPATTU", "THE NILGIRIS", ...],
  "count": 39
}
```

### 4. Get Hierarchical Data
```
GET /dropdown/hierarchical
```
**Response:**
```json
{
  "success": true,
  "hierarchical": {
    "TAMILNADU": {
      "districts": ["ARIYALUR", "THE NILGIRIS", ...],
      "district_count": 39
    },
    ...
  },
  "total_states": 37,
  "total_districts": 751
}
```

### 5. Get All Taluks (NEW!)
```
GET /dropdown/taluks
```
**Response:**
```json
{
  "success": true,
  "taluks": ["Ajjampura", "Alur", "Ankola", ...],
  "count": 1203
}
```

### 6. Get Taluks by State (NEW!)
```
GET /dropdown/taluks/{state}
```
**Example:** `/dropdown/taluks/KARNATAKA`

**Response:**
```json
{
  "success": true,
  "state": "KARNATAKA",
  "taluks": ["Ajjampura", "Alur", "Ankola", ...],
  "count": 942
}
```

### 7. Get Taluks by District (NEW!)
```
GET /dropdown/taluks/{state}/{district}
```
**Example:** `/dropdown/taluks/KARNATAKA/Chikkamagaluru`

**Response:**
```json
{
  "success": true,
  "state": "KARNATAKA",
  "district": "Chikkamagaluru",
  "taluks": ["Ajjampura", "Chikmagalur", "Kadur", "Kalasa", "Koppa", "Mudigere", "Narasimharajapura", "Sringeri", "Tarikere"],
  "count": 9
}
```

### 8. Get Enhanced Hierarchical Data (NEW!)
```
GET /dropdown/enhanced-hierarchical
```
**Response:**
```json
{
  "success": true,
  "hierarchical": {
    "KARNATAKA": {
      "districts": ["Chikkamagaluru", "Bengaluru (Urban)", ...],
      "district_count": 31,
      "taluks": ["Ajjampura", "Alur", "Ankola", ...],
      "taluk_count": 942
    },
    ...
  },
  "total_states": 37,
  "total_districts": 751,
  "total_taluks": 1203
}
```

### 9. Get Taluk Data (NEW!)
```
GET /dropdown/taluk-data/{state}/{district}/{taluk}
```
**Example:** `/dropdown/taluk-data/KARNATAKA/Chikkamagaluru/Ajjampura`

**Response:**
```json
{
  "success": true,
  "state": "KARNATAKA",
  "district": "Chikkamagaluru",
  "taluk": "Ajjampura",
  "data": {
    "stage_of_ground_water_extraction": "146.97",
    "categorization": "over_exploited",
    "rainfall_mm": "638.84",
    "ground_water_recharge_ham": "4485.3",
    "ground_water_extraction_ham": "5932.73",
    "pre_monsoon_trend": "Rising",
    "post_monsoon_trend": "Rising",
    "quality_tagging": "[0.0]",
    "assessment_unit": "N/A",
    "year": "2021"
  }
}
```

## Frontend Components

### LocationDropdown Component
**File:** `frontend/src/components/LocationDropdown.jsx`

**Props:**
- `onLocationSelect(state, district)` - Callback when location is selected
- `selectedState` - Currently selected state
- `selectedDistrict` - Currently selected district

**Features:**
- Automatic state loading on mount
- Dynamic district loading based on state selection
- Loading states and error handling
- Responsive design

### DropdownDemo Page
**File:** `frontend/src/pages/DropdownDemo.jsx`

**Features:**
- Complete dropdown demonstration
- Real-time groundwater analysis
- Integration with existing API
- User-friendly interface

## Usage Examples

### 1. Basic Dropdown Usage
```jsx
import LocationDropdown from '../components/LocationDropdown';

function MyComponent() {
  const [selectedState, setSelectedState] = useState('');
  const [selectedDistrict, setSelectedDistrict] = useState('');

  const handleLocationSelect = (state, district) => {
    setSelectedState(state);
    setSelectedDistrict(district);
    
    if (state && district) {
      // Perform analysis
      analyzeGroundwater(state, district);
    }
  };

  return (
    <LocationDropdown
      onLocationSelect={handleLocationSelect}
      selectedState={selectedState}
      selectedDistrict={selectedDistrict}
    />
  );
}
```

### 2. API Integration
```javascript
// Get all states
const response = await fetch('/dropdown/states');
const data = await response.json();
const states = data.states;

// Get districts for Tamil Nadu
const response = await fetch('/dropdown/districts/TAMILNADU');
const data = await response.json();
const districts = data.districts;
```

## Data Structure

### States (37 total)
- ANDAMAN AND NICOBAR ISLANDS
- ANDHRA PRADESH
- ARUNACHAL PRADESH
- ASSAM
- BIHAR
- CHANDIGARH
- CHHATTISGARH
- DADRA AND NAGAR HAVELI
- DAMAN AND DIU
- DELHI
- GOA
- GUJARAT
- HARYANA
- HIMACHAL PRADESH
- JAMMU AND KASHMIR
- JHARKHAND
- KARNATAKA
- KERALA
- LADAKH
- LAKSHDWEEP
- MADHYA PRADESH
- MAHARASHTRA
- MANIPUR
- MEGHALAYA
- MIZORAM
- NAGALAND
- ODISHA
- PUDUCHERRY
- PUNJAB
- RAJASTHAN
- SIKKIM
- TAMILNADU
- TELANGANA
- TRIPURA
- UTTAR PRADESH
- UTTARAKHAND
- WEST BENGAL

### Key Districts
- **THE NILGIRIS** (Tamil Nadu) - Ooty/Udhagamandalam
- **BANGALORE URBAN** (Karnataka) - Bengaluru
- **MUMBAI** (Maharashtra) - Mumbai
- **CHENNAI** (Tamil Nadu) - Chennai
- **HYDERABAD** (Telangana) - Hyderabad

### Key Taluks (Examples)
- **Chikkamagaluru District (Karnataka):**
  - Ajjampura (Over-exploited: 146.97% extraction)
  - Koppa (Safe: 27.73% extraction)
  - Narasimharajapura (Safe: 19.81% extraction)
  - Chikmagalur (Safe: 42.19% extraction)
- **Bengaluru (Urban) District (Karnataka):**
  - Bangalore-East (Over-exploited: 378.85% extraction)
- **Chikkaballapura District (Karnataka):**
  - Gauribidanur (Over-exploited: 165.34% extraction)

## Performance
- **States Loading:** < 0.1 seconds
- **Districts Loading:** < 0.1 seconds
- **Taluks Loading:** < 0.1 seconds
- **API Response Size:** ~5KB for states, ~9KB for districts, ~15KB for taluks
- **Memory Usage:** Minimal (data cached in browser)
- **Enhanced Data Loading:** ~0.2 seconds (first time only)

## Error Handling
- Network connection errors
- Invalid state/district selections
- Server errors (500 status codes)
- Data loading failures

## Testing
Run the test suite:
```bash
python test_dropdown_integration.py
python test_dropdown_api.py
```

## Integration with Existing System
The dropdown system integrates seamlessly with:
- ✅ **Location Synonyms** (Ooty → THE NILGIRIS)
- ✅ **Groundwater Analysis API**
- ✅ **Existing Frontend Components**
- ✅ **Main2.py Backend**

## Future Enhancements
- [x] Add taluk/block level selection (COMPLETED!)
- [ ] Add search/filter functionality
- [ ] Add map integration
- [ ] Add recent selections history
- [ ] Add favorites/bookmarks
- [ ] Add block/mandal/village level selection
- [ ] Add taluk-level groundwater analysis

## Troubleshooting

### Common Issues
1. **Dropdown not loading:** Check if server is running on port 8000
2. **Districts not showing:** Verify state selection is valid
3. **Ooty not found:** Ensure "THE NILGIRIS" is selected in Tamil Nadu
4. **API errors:** Check network connection and server logs

### Debug Commands
```bash
# Test API endpoints
curl http://localhost:8000/dropdown/states
curl http://localhost:8000/dropdown/districts/TAMILNADU

# Test data integrity
python test_dropdown_integration.py
```

## Support
For issues or questions about the dropdown system, check:
1. Server logs for API errors
2. Browser console for frontend errors
3. Network tab for API call failures
4. Test scripts for data validation
