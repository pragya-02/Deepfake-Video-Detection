import { MdOutlineMail } from "react-icons/md";
function Footer() {
  return (
    <section className="w-full bg-background p-8 pt-80 flex">
      <div className="w-1/2 flex items-center justify-center">
        <img src="logo.png" alt="" className="scale-50" />
      </div>
      <div className="w-1/2 flex items-center  justify-center">
        <div className="flex flex-col items-center justify-center gap-6">
          <h1 className="text-text font-bold text-[1.4rem] font-body">
            Contact Us
          </h1>
          <div className="flex flex-col gap-2 items-center justify-center">
            <h1 className="text-text font-body text-[1.2rem]">
              Khwopa College of Engineering
            </h1>
            <h1 className="text-text font-body text-[1rem]">
              Libali, Bhaktapur
            </h1>
          </div>

          <a
            href="mailto:deepfakedetectionprojectkhwopa@gmail.com"
            className="flex items-center gap-2 text-text text-[1rem] font-body hover:text-primary"
          >
            <MdOutlineMail />
            <h1>Deepfakedetectionprojectkhwopa@gmail.com</h1>
          </a>
        </div>
      </div>
    </section>
  );
}
export default Footer;
