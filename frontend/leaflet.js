var map = L.map('map', {
    minZoom: 0,
    maxZoom: 10
}).setView([13.0827, 80.2707], 10);
map.locate({setView: true, maxZoom: 10});

function onLocationFound(e) {
    var radius = e.accuracy;
    alert("Location detected!");
    L.marker(e.latlng).addTo(map)
        .bindPopup("You are within " + radius + " meters from this point").openPopup();

    L.circle(e.latlng, radius).addTo(map);
}

var heat = L.heatLayer([
    // Chennai city
    [13.0827, 80.2707, 0.90],
    [13.0674, 80.2376, 0.85],
    [13.0604, 80.2496, 0.80],
    [13.0475, 80.2090, 0.75],
    [13.1143, 80.2948, 0.88],
    [13.1067, 80.2433, 0.82],
    [13.0358, 80.2619, 0.78],
    [13.0108, 80.2340, 0.72],
    [13.0213, 80.2221, 0.70],
    [13.0418, 80.1565, 0.68],

    // North Chennai
    [13.1482, 80.2825, 0.92],
    [13.1350, 80.2930, 0.95],
    [13.1230, 80.2900, 0.90],
    [13.1600, 80.3000, 0.88],
    [13.1750, 80.3050, 0.83],
    [13.1900, 80.3100, 0.80],

    // West Chennai
    [13.0730, 80.1900, 0.65],
    [13.0850, 80.1700, 0.62],
    [13.0950, 80.1500, 0.58],
    [13.1100, 80.1350, 0.55],
    [13.1250, 80.1200, 0.52],
    [13.0500, 80.1800, 0.70],

    // South Chennai
    [13.0200, 80.2600, 0.72],
    [13.0000, 80.2500, 0.68],
    [12.9800, 80.2400, 0.65],
    [12.9600, 80.2300, 0.60],
    [12.9400, 80.2200, 0.55],
    [12.9200, 80.2100, 0.50],

    // OMR / Sholinganallur
    [12.9800, 80.2200, 0.62],
    [12.9500, 80.2300, 0.65],
    [12.9300, 80.2300, 0.68],
    [12.9000, 80.2300, 0.70],
    [12.8700, 80.2200, 0.64],
    [12.8500, 80.2100, 0.60],

    // Tambaram
    [12.9249, 80.1000, 0.58],
    [12.9400, 80.1200, 0.60],
    [12.9500, 80.1000, 0.62],
    [12.9700, 80.1000, 0.55],
    [12.9000, 80.1200, 0.52],

    // Chengalpattu direction
    [12.7000, 79.9800, 0.45],
    [12.6800, 79.9900, 0.48],
    [12.6500, 79.9800, 0.42],
    [12.6200, 79.9700, 0.40],

    // Kanchipuram direction
    [12.8350, 79.7000, 0.50],
    [12.8500, 79.7300, 0.52],
    [12.8700, 79.7500, 0.55],
    [12.9000, 79.7800, 0.57],

    // Tiruvallur direction
    [13.1400, 80.0000, 0.48],
    [13.1600, 79.9500, 0.45],
    [13.1700, 79.9000, 0.42],
    [13.1200, 79.9500, 0.50],

    // Additional points around Chennai
    [13.2000, 80.2500, 0.76],
    [13.2200, 80.2700, 0.72],
    [13.1000, 80.3300, 0.82],
    [13.0800, 80.3500, 0.78],
    [13.0500, 80.3400, 0.75],
    [13.0000, 80.3200, 0.70],
    [12.9600, 80.3000, 0.68],
    [12.9200, 80.2800, 0.64],
    [12.8800, 80.2700, 0.60]

], {
    radius: 30,
    blur: 20,
    maxZoom: 12,
    max: 1.0
}).addTo(map);

function onLocationError(e) {
    alert(e.message);
}

map.on('locationerror', onLocationError);

map.on('locationfound', onLocationFound);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

//svar marker = L.marker([13.10088, 80.249634]).addTo(map);

var circle = L.circle([20.94092, 81.782227], {
    color: 'red',
    fillColor: '#f03',
    fillOpacity: 0.5,
    radius: 111500
}).addTo(map);

circle.bindPopup("Risk Score: 99");


function onMapClick(e) {
    var popup = L.popup()
    .setLatLng(e.latlng)
    .setContent(e.latlng.toString())
    .openOn(map);
}

map.on('click', onMapClick);