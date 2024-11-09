import './app.css';
import Alpine from 'alpinejs';
import axios from 'axios';
import { Chart, registerables } from 'chart.js';

window.axios = axios;
window.Alpine = Alpine;
window.Chart = Chart;

Chart.register(...registerables);

Alpine.start();
