#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <functional> 
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Scalar = double; //must match VectorXd type

class ExactDiscreteModulusOfContinuity {

    // implement by default the EUCLIDEAN distance (use l2 norms)
    
    public:
        ExactDiscreteModulusOfContinuity(){};

    void init(const Matrix &P, const Matrix &f, std::string dx_type = "EUCLIDEAN", const std::string dy_type = "EUCLIDEAN"){
        //P is dxn dimensional matrix (n datapoints, d dimensions)
        //f is qxn dimensional matrix (n datapoints, q dimensions) 
        //where f.col(i) contains function values for P.col(i)
        

        setDistanceType(dx, dx_type);
        setDistanceType(dy, dy_type);
        
        //determine the max_distance (we will compute moc for t= 0,...,max_distance)
        max_distance = getMaxDistance(dx, P);
        

    }

    Scalar computeMoc(const Matrix &P, const Matrix &f, const Scalar &d){
        //this will make use of dx, dy initialized in init (we might cast the getMax function and re-use it here)
        //computes moc exactly, for freely chosen delta value t
        Scalar d_max=0;

        #pragma omp parallel for reduction(max : d_max)
        for (int i=0; i < P.cols();i++) {
            for (int j=0; j<i;j++){
                Scalar dis_x = dx(P.col(j),P.col(i));
                if(dis_x<=d){
                    Scalar dis_y = dy(f.col(j),f.col(i));
                    if (dis_y>d_max){
                        d_max = dis_y;
                    }
                       
                }

            }

        }
        return d_max;
    }

    Vector computeMocPlot(const Matrix &P, const Matrix &f, const Scalar &d){
        //assumes the delta_steps are evenly spaced (0, d, 2d,...)
        //d is the delta step in moc computation
        int T = static_cast<int>(std::ceil((max_distance)/d))+1;  //(guarantee an integer) 
        Vector t_values(T);
        Vector mocplot(T);
        Vector B = Vector::Zero(T);; 

        mocplot(0)=0;
        t_values(0)=0;

        for(int i=1; i<T;i++){
            t_values(i)=t_values(i-1)+d;
        }

        //first compute all pairwise distances
        int num_pairs = P.cols()*(P.cols()+1)/2;
        Vector vec_P(num_pairs);
        Vector vec_f(num_pairs);

        #pragma omp parallel for
        for (int i=0; i < P.cols();i++) {
            for (int j=i; j< P.cols();j++){
                int idx = i * P.cols() - (i * (i-1))/2 + (j - i);
                Scalar dist_x = dx(P.col(j),P.col(i));
                Scalar dist_y = dy(f.col(j),f.col(i));
                
                vec_P(idx)=dist_x;
                vec_f(idx)=dist_y;
                
            }

        }

        //we discretize all pairwise distances into T bins of width d (the zero bin, contains zero distances)
        //e.g. d_x(i,j) belongs to bin z if zd<d_x(i,j)/d<=(z+1)d

        //Thread-local maxima: reduce across threads with element-wise max.
        for(int i=0; i<num_pairs;i++){

            // if (vec_P(i) <= d) {
            //     bin_index =0;
            // } else {
            //     bin_index = static_cast<int>(std::ceil(vec_P(i) / d)) - 1;
            // }
            int bin_index = static_cast<int>(std::ceil(vec_P(i) / d));
            
            
            B(bin_index) = std::max(B(bin_index),vec_f(i)); // at the end it will contain max f-distance for this bin (e.g. across  all points falling in i-th bin)
        }


        
        for(int i=1; i<T;i++){
            mocplot(i) = std::max(mocplot(i-1), B(i));
        }
       

        return mocplot;
    }




    private:
        std::function<Scalar(const Vector &, const Vector &)> dx;
        std::function<Scalar(const Vector &, const Vector &)> dy;
        Scalar max_distance;

        void setDistanceType(std::function<Scalar(const Vector &, const Vector &)> &df, const std::string &dist_type) {

            if (dist_type == "EUCLIDEAN") {
            df = [](const Vector &x, const Vector &y) {
                return (x - y).norm();
            };
            } else
            assert(false && "desired distance not implemented");
            return;
        }




        Scalar getMaxDistance(const std::function<Scalar(const Vector &, const Vector &)> &df, const Matrix &P){
            //assuming points are stored as columns in P, df is the distance function to be used.
            Scalar max_d=0;

            #pragma omp parallel for reduction(max : max_d)
            for (int i=0; i < P.cols();i++) {
                
                for (int j=0; j<i;j++){
                    Scalar dist = df(P.col(j),P.col(i));
                    if(dist>max_d){
                        max_d = dist;
                    }

                }

            }
            return max_d;
        }
    
};


//define the module for python import (exactdmoc)
PYBIND11_MODULE(exactdmoc, m){
    py::class_<ExactDiscreteModulusOfContinuity>(m, "ExactDiscreteModulusOfContinuity")
        .def(py::init<>())
        .def("init", &ExactDiscreteModulusOfContinuity::init,
             py::arg("P"), py::arg("f"),
             py::arg("dx_type") = "EUCLIDEAN",
             py::arg("dy_type") = "EUCLIDEAN")
        .def("computeMoc", &ExactDiscreteModulusOfContinuity::computeMoc)
        .def("computeMocPlot", &ExactDiscreteModulusOfContinuity::computeMocPlot,
             py::arg("P"), py::arg("f"), py::arg("d"));
}



// int main(){


//     Matrix x(2,2);
//     Matrix f(2,2);

//     x << 0,1,
//          2,3;
//     f << 0,1,
//          3,2;
    

//     ExactDiscreteModulusOfContinuity dmoc;
//     dmoc.init(x,f,"EUCLIDEAN","EUCLIDEAN");
//     std::cout << dmoc.computeMocPlot(x, f, 0.5);

// }

