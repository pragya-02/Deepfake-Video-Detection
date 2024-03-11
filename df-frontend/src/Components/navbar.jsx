import { FaGoogle } from "react-icons/fa";
import { FaUser } from "react-icons/fa";
import React, { useState, useEffect } from 'react';
import { googleLogout, useGoogleLogin } from '@react-oauth/google';
import axios from 'axios';


function Navbar({reload}) {



  const [ user, setUser ] = useState(false);
  const [ profile, setProfile ] = useState(false);
  
  const login = useGoogleLogin({
      onSuccess: (codeResponse) => setUser(codeResponse),
      onError: (error) => console.log('Login Failed:', error)
  });
  


  useEffect(() => {
    const storedAuthStatus = localStorage.getItem('user');
    if (storedAuthStatus) {
      setProfile(JSON.parse(storedAuthStatus));
    }
  }, []);

  useEffect(
      () => {
          if (user ) {
              axios
                  .get(`https://www.googleapis.com/oauth2/v1/userinfo?access_token=${user.access_token}`, {
                      headers: {
                          Authorization: `Bearer ${user.access_token}`,
                          Accept: 'application/json'
                      }
                  })
                  .then((res) => {
                      setProfile(res.data);
                      localStorage.setItem('user', JSON.stringify(res.data));
                      reload(true);
                      
                  })
                  .catch((err) => console.log(err));
          }
      },
      [ user]
  );
  
  // log out function to log the user out of google and set the profile array to null
  const logout = () => {
      googleLogout();
      setProfile(null);
      localStorage.setItem('user', JSON.stringify(false));
      reload(true);
  };
  
  
  


  return (
    <header className="bg-background  flex justify-between items-center p-4 border-b-[2px] border-border">
    <a href="/" className="flex justify-center items-center">
      <img className="mx-2" src="logo-college.png" height={50} width={50} alt="" />
      <h1 className="text-text text-xl font-heading">Khwopa College of Engineering</h1>
    </a>
    <nav className="flex space-x-4 font-heading">
      {profile ? <button onClick={logout} className="bg-transparent flex items-center text-text hover:drop-shadow-[0_35px_35px_rgba(58,49,216,0.5)] hover:bg-primary font-heading hover:text-text mx-8 py-2 px-4 border-[2px] border-border hover:border-transparent rounded-lg">
      <FaUser />  &nbsp;  &nbsp;
  {profile.name}
</button> :
<button onClick={login} className="bg-transparent flex items-center text-text hover:drop-shadow-[0_35px_35px_rgba(58,49,216,0.5)] hover:bg-primary font-heading hover:text-text mx-8 py-2 px-4 border-[2px] border-border hover:border-transparent rounded-lg">
    <FaGoogle />
 &nbsp; Sign In
</button> 
}
    
     
    
    </nav>
  </header>
  );
}
export default Navbar;
