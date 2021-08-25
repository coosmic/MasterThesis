
#include "definitions.h"
#include "configuration.h"
#include "debug.h"
//#include "registration_cgal.h"
#include "test.h"
#include "registration_pcl.h"
#include "utilities.h"
#include "utilities_io.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>


namespace np = boost::python::numpy;

char const* greet()
{
   return "hello, python world. Greetings from cpp";
}

BOOST_PYTHON_MODULE(pgm_py_binding)
{
    using namespace boost::python;
    //Py_Initialize();
    np::initialize();
    def("greet", greet);

    // Write the bindings here
}