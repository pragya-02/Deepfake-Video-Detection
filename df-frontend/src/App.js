import './App.css';
import Navbar from './Components/navbar'
import Body from "./Components/Upload/Body";
import Footer from './Components/footer';
import { useState } from 'react';

function App() {
const [data, setData] = useState(JSON.parse(localStorage.getItem('user')));

  const reload_data = (data) => {
    setData(JSON.parse(localStorage.getItem('user')));
  }


  
  return (
    <div className="App">
   
    <Navbar reload={reload_data}/>
    <Body profile={data}/>
    <Footer/>
    </div>
  );
}

export default App;
