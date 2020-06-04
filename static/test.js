
<script type="text/javascript">
var ctx = document.getElementById('myChart').getContext('2d');
var chart = new Chart(ctx, {
    // The type of chart we want to create
    type: 'bar',

    // The data for our dataset
    data: {
        labels: ['Setosa', 'Versicolor', 'Virginica'],
        datasets: [{
            label:'Iris flower',
            //backgroundColor: 'rgba(0,255, 255, 0.1)',
            //backgroundColor:['#f1c40f','#e67e22','#2980b9'],
            backgroundColor: [ 'rgba(255, 99, 132, 0.2)',
                  'rgba(54, 162, 235, 0.2)', 'rgba(255, 159, 64, 0.2)'],
          //  borderColor: 'rgb(255, 99, 132)',
          borderColor:[  'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 159, 64, 1)'],
             borderWidth: 1,
            data: [100, 20, 15]
        }]
    },

    // Configuration options go here
    options: {
      title: {
            display: true,
            text: 'Iris Species Bar graph'
        },
         scales: {
             yAxes: [{
                 ticks: {
                     beginAtZero: true
                 }
             }]
         }
     }
});
</script>
